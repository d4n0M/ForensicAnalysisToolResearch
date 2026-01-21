import os
import cv2
import time
import torch
import numpy as np
import shutil
from tqdm import tqdm
from autodistill_grounded_sam import GroundedSAM
from autodistill.detection import CaptionOntology

def process_video_on_gpu(video_path, gpu_id, queue):
    torch.cuda.set_device(gpu_id)
    start_time = time.time()

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    tmp_dir = os.path.join("detections", f"{video_name}_frames")
    os.makedirs(tmp_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[GPU {gpu_id}] ❌ Could not open {video_path}")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 15  # fallback for WEBM
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[GPU {gpu_id}] Processing {video_name} ({frame_count} frames @ {fps:.2f} FPS)")

    ontology = CaptionOntology({"there is a gun": "gun"})
    model = GroundedSAM(ontology=ontology)

    frame_idx = 0
    for _ in tqdm(range(frame_count), desc=f"[GPU {gpu_id}] Inference {video_name}", position=gpu_id):
        ret, frame = cap.read()
        if not ret or frame is None:
            break
        try:
            results = model.predict(frame)

            if results is not None and hasattr(results, "xyxy") and hasattr(results, "class_id"):
                boxes = results.xyxy
                labels = [model.ontology.classes()[i] for i in results.class_id]

                for box, label in zip(boxes, labels):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                # No detections or invalid result
                pass

        except Exception as e:
            print(f"[GPU {gpu_id}] ⚠️ Frame {frame_idx} failed: {e}")
            continue

        tmp_img_path = os.path.join(tmp_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(tmp_img_path, frame)
        frame_idx += 1

    cap.release()

    # Reconstruct video
    output_path = os.path.join("detections", f"{video_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_files = sorted(os.listdir(tmp_dir))
    for f in tqdm(frame_files, desc=f"[GPU {gpu_id}] Reconstruct {video_name}", position=gpu_id+10):
        frame_path = os.path.join(tmp_dir, f)
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)

    out.release()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    elapsed = time.time() - start_time
    print(f"[GPU {gpu_id}] ✅ Done {video_name} in {elapsed:.2f}s ({frame_idx} frames)")
    queue.put((video_name, elapsed))

import os
import torch
import torch.multiprocessing as mp

def gpu_worker(gpu_id, videos, queue):
    """Worker entrypoint for a given GPU.

    Args:
        gpu_id (int): CUDA device id to use.
        videos (list[str]): List of video file paths assigned to this GPU.
        queue (mp.Queue): Multiprocessing queue used to report timings.
    """
    # process each assigned video sequentially on this GPU
    for video_path in videos:
        process_video_on_gpu(video_path, gpu_id, queue)

def main():
    mp.set_start_method("spawn", force=True)

    VIDEO_DIR = "../"
    NUM_GPUS = torch.cuda.device_count() - 2
    all_videos = [os.path.join(VIDEO_DIR, f) for f in os.listdir(VIDEO_DIR) if f.lower().endswith(".mp4")]

    if not all_videos:
        print("❌ No videos found.")
        return

    print(f"🧠 Detected {NUM_GPUS} GPUs, distributing {len(all_videos)} videos...")

    # Distribute videos evenly
    chunks = [[] for _ in range(NUM_GPUS)]
    for i, vid in enumerate(all_videos):
        chunks[i % NUM_GPUS].append(vid)

    queue = mp.Queue()
    processes = []

    for gpu_id in range(NUM_GPUS):
        if not chunks[gpu_id]:
            continue
        p = mp.Process(target=gpu_worker, args=(gpu_id, chunks[gpu_id], queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    total_times = []
    while not queue.empty():
        total_times.append(queue.get())

    print("\n📊 Timing Summary:")
    for name, t in total_times:
        print(f" - {name}: {t:.2f}s")

    print("\n✅ All videos processed successfully.")

if __name__ == "__main__":
    main()

