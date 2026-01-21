import os
import cv2
import time
import torch
import shutil
import numpy as np
from tqdm import tqdm
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection
from transformers import Owlv2Processor, Owlv2ForObjectDetection
from torchvision.ops import nms
import torch.multiprocessing as mp
import gc


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
        fps = 15
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[GPU {gpu_id}] Processing {video_name} ({frame_count} frames @ {fps:.2f} FPS)")

    processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
    model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble").to(f"cuda:{gpu_id}").eval()

    classes = ["a photo of a gun", "a photo of a weapon"]

    frame_idx = 0
    progress = tqdm(total=frame_count, desc=f"[GPU {gpu_id}] {video_name}", position=gpu_id)

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            inputs = processor(text=classes, images=image, return_tensors="pt").to(f"cuda:{gpu_id}")

            with torch.no_grad():
                outputs = model(**inputs)

            target_sizes = torch.Tensor([image.size[::-1]]).to(f"cuda:{gpu_id}")
            results = processor.post_process_object_detection(
                outputs=outputs, target_sizes=target_sizes, threshold=0.3
            )[0]

            # Initial score filtering
            mask = results["scores"] > 0.4
            boxes = results["boxes"][mask]
            scores = results["scores"][mask]
            labels = results["labels"][mask]

            # Apply Non-Maximum Suppression (NMS) to remove duplicate detections
            if len(boxes) > 0:
                # NMS requires boxes, scores - it returns indices to keep
                keep_indices = nms(boxes, scores, iou_threshold=0.3)
                
                boxes = boxes[keep_indices]
                scores = scores[keep_indices]
                labels = labels[keep_indices]

            # Draw remaining boxes after NMS
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = map(int, box.tolist())
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"{classes[label]}: {score:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            del outputs, inputs, results
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"[GPU {gpu_id}] ⚠️ Frame {frame_idx} failed: {e}")

        tmp_img_path = os.path.join(tmp_dir, f"frame_{frame_idx:06d}.jpg")
        cv2.imwrite(tmp_img_path, frame)
        frame_idx += 1
        progress.update(1)

        if frame_idx % 200 == 0:
            gc.collect()
            torch.cuda.empty_cache()

    progress.close()
    cap.release()

    output_path = os.path.join("detections", f"{video_name}_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_files = sorted(os.listdir(tmp_dir))
    for f in tqdm(frame_files, desc=f"[GPU {gpu_id}] Reconstruct {video_name}", position=gpu_id + 10):
        frame_path = os.path.join(tmp_dir, f)
        frame = cv2.imread(frame_path)
        if frame is not None:
            out.write(frame)
    out.release()

    shutil.rmtree(tmp_dir, ignore_errors=True)
    elapsed = time.time() - start_time
    print(f"[GPU {gpu_id}] ✅ Done {video_name} in {elapsed:.2f}s ({frame_idx} frames)")
    queue.put((video_name, elapsed))


def gpu_worker(gpu_id, videos, queue):
    gpu_id = gpu_id 
    for video_path in videos:
        process_video_on_gpu(video_path, gpu_id, queue)


def main():
    mp.set_start_method("spawn", force=True)
    VIDEO_DIR = "."
    os.makedirs("detections", exist_ok=True)

    NUM_GPUS = torch.cuda.device_count()
    all_videos = [os.path.join(VIDEO_DIR, f)
                  for f in os.listdir(VIDEO_DIR)
                  if f.lower().endswith(".mp4")]

    if not all_videos:
        print("❌ No videos found.")
        return

    print(f"🧠 Detected {NUM_GPUS} GPUs, distributing {len(all_videos)} videos...")

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