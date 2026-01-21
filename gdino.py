#!/usr/bin/env python3
import os
import time
import gc
import shutil
import cv2
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torch.multiprocessing as mp

# -------------------------
# Configuration
# -------------------------
MODEL_ID = "IDEA-Research/grounding-dino-base"
VIDEO_DIR = "../"           # folder to scan for .mp4
OUTPUT_DIR = "detections"   # folder where annotated mp4s will be saved
os.makedirs(OUTPUT_DIR, exist_ok=True)

FRAME_SKIP = 1              # process every Nth frame
TEXT_THRESHOLD = 0.5
CONF_THRESHOLD = 0.3
CLASSES = ["gun", "weapon", "knife"]
TEXT = " ".join([f"a {c}." for c in CLASSES])

GC_EVERY_N_FRAMES = 200     # periodic cleanup
QUEUE_TIMEOUT = 300         # seconds to wait for a worker result

# -------------------------
# Worker: process a list of full videos on one GPU (streaming)
# -------------------------
def process_videos_on_gpu(gpu_id, video_list, queue):
    # Bind to GPU and load model once for this worker
    torch.cuda.set_device(gpu_id)
    device_str = f"cuda:{gpu_id}"

    try:
        processor = AutoProcessor.from_pretrained(MODEL_ID)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(MODEL_ID).to(device_str).eval()
    except Exception as e:
        print(f"[GPU {gpu_id}] ❌ Failed to load model: {e}")
        # Report failures for all assigned videos
        for video_path in video_list:
            queue.put((os.path.splitext(os.path.basename(video_path))[0], False, 0.0))
        return

    # font fallback
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except Exception:
        font = ImageFont.load_default()

    for video_path in video_list:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        t0 = time.time()

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[GPU {gpu_id}] ❌ Could not open {video_path}")
            queue.put((video_name, False, 0.0))
            continue

        # Video properties
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        print(f"[GPU {gpu_id}] Processing {video_name} ({total_frames} frames @ {fps:.2f} FPS)")

        # Prepare VideoWriter for annotated output
        out_path = os.path.join(OUTPUT_DIR, f"{video_name}_annotated.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))
        if not out_writer.isOpened():
            print(f"[GPU {gpu_id}] ❌ Could not open VideoWriter for {out_path}")
            cap.release()
            queue.put((video_name, False, 0.0))
            continue

        frame_idx = 0
        saved = 0
        pbar_desc = f"[GPU {gpu_id}] {video_name}"
        pbar = tqdm(total=total_frames if total_frames > 0 else None, desc=pbar_desc, position=gpu_id, leave=False)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # Optionally skip frames
            if frame_idx % FRAME_SKIP == 0:
                try:
                    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                    inputs = processor(images=pil, text=TEXT, return_tensors="pt").to(device_str)
                    with torch.no_grad():
                        outputs = model(**inputs)

                    # Post-process
                    results = processor.post_process_grounded_object_detection(
                        outputs,
                        inputs.input_ids,
                        text_threshold=TEXT_THRESHOLD,
                        target_sizes=[pil.size[::-1]]
                    )[0]

                    draw = ImageDraw.Draw(pil)
                    for box, score, label in zip(
                        results.get("boxes", []),
                        results.get("scores", []),
                        results.get("text_labels", [])
                    ):
                        if score is None:
                            continue
                        if float(score) < CONF_THRESHOLD:
                            continue
                        b = [int(x) for x in box.tolist()]
                        x1, y1, x2, y2 = b
                        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
                        draw.text((x1, max(0, y1 - 16)), f"{label}: {float(score):.2f}", fill="red", font=font)

                    annotated = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
                    out_writer.write(annotated)
                    saved += 1

                    # free intermediate tensors
                    del inputs, outputs, results, pil

                except Exception as e:
                    print(f"[GPU {gpu_id}] ⚠️ Frame {frame_idx} failed: {e}")
                    # on error, still attempt to write the original frame so timing aligns
                    try:
                        out_writer.write(frame)
                    except Exception:
                        pass
            else:
                # skipped frame: write original
                out_writer.write(frame)

            frame_idx += 1
            pbar.update(1)

            # periodic cleanup
            if frame_idx % GC_EVERY_N_FRAMES == 0:
                gc.collect()
                torch.cuda.empty_cache()

        pbar.close()
        cap.release()
        out_writer.release()

        elapsed = time.time() - t0
        print(f"[GPU {gpu_id}] ✅ Done {video_name} — saved {saved} annotated frames in {elapsed:.1f}s")
        queue.put((video_name, True, elapsed))

    # end for videos
    # final cleanup
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()


# -------------------------
# Main controller
# -------------------------
def main():
    mp.set_start_method("spawn", force=True)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        print("[MAIN] No GPUs detected — aborting.")
        return

    # discover mp4 files
    all_videos = [
        os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR)
        if f.lower().endswith(".mp4")
    ]
    if not all_videos:
        print("[MAIN] No MP4 files found in", VIDEO_DIR)
        return

    print(f"[MAIN] Found {len(all_videos)} videos, using {num_gpus} GPUs (round-robin assignment).")

    # distribute full videos round-robin across GPUs
    chunks = [[] for _ in range(num_gpus)]
    for i, v in enumerate(all_videos):
        chunks[i % num_gpus].append(v)

    queue = mp.Queue()
    processes = []

    for gpu_id in range(num_gpus):
        if not chunks[gpu_id]:
            continue
        p = mp.Process(target=process_videos_on_gpu, args=(gpu_id, chunks[gpu_id], queue))
        p.start()
        processes.append(p)

    # collect results
    results = []
    jobs_total = len(all_videos)
    got = 0
    while got < jobs_total:
        try:
            name, ok, elapsed = queue.get(timeout=QUEUE_TIMEOUT)
            got += 1
            results.append((name, ok, elapsed))
            print(f"[MAIN] Completed {got}/{jobs_total}: {name} (ok={ok})")
        except Exception:
            alive = any(p.is_alive() for p in processes)
            if not alive:
                print("[MAIN] Workers not alive and queue timed out — exiting.")
                break
            else:
                print("[MAIN] Waiting for workers to produce results...")

    # join/terminate processes
    for p in processes:
        p.join(timeout=10)
        if p.is_alive():
            p.terminate()

    # summary
    print("\n[MAIN] Summary:")
    for name, ok, elapsed in results:
        print(f" - {name}: ok={ok}, {elapsed:.1f}s")

    print("[MAIN] All done.")

if __name__ == "__main__":
    main()

