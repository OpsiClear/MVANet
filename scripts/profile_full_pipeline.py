"""Profile the full inference pipeline including I/O."""

import torch
import numpy as np
from pathlib import Path
import time
import sys
import cv2

# Safe output for Windows
class SafeWriter:
    def __init__(self, stream):
        self.stream = stream
    def write(self, s):
        try:
            self.stream.write(s)
        except UnicodeEncodeError:
            self.stream.write(s.encode('ascii', 'replace').decode('ascii'))
    def flush(self):
        self.stream.flush()

sys.stdout = SafeWriter(sys.stdout)
sys.stderr = SafeWriter(sys.stderr)

from oc_masker_mvanet import MVANetModel


def profile_full():
    """Profile each stage of the full pipeline."""
    device = torch.device("cuda:0")

    # Use actual test images
    test_folder = Path("D:/test/scan_20250928_115837_bella_mushroom/images/cam_1")
    image_paths = list(test_folder.glob("*.png"))[:20]

    print(f"Profiling with {len(image_paths)} images from {test_folder.name}")

    # Create model
    model = MVANetModel(trt_batch_size=1)
    model.load(None, device)
    model.optimize_for_inference(device)

    # Warm up
    img = cv2.imread(str(image_paths[0]))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    t, m = model.preprocess(img_rgb)
    out = model.forward(t)
    mask = model.postprocess(out, m)
    torch.cuda.synchronize()

    num_runs = 3

    # Profile each stage
    load_times = []
    cvt_times = []
    preprocess_times = []
    forward_times = []
    postprocess_times = []
    save_mask_times = []
    save_overlay_times = []

    for run in range(num_runs):
        for img_path in image_paths:
            # 1. Load image
            start = time.perf_counter()
            img = cv2.imread(str(img_path))
            load_times.append(time.perf_counter() - start)

            # 2. Color conversion
            start = time.perf_counter()
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cvt_times.append(time.perf_counter() - start)

            # 3. Preprocess
            start = time.perf_counter()
            tensor, meta = model.preprocess(img_rgb)
            torch.cuda.synchronize()
            preprocess_times.append(time.perf_counter() - start)

            # 4. Forward
            start = time.perf_counter()
            output = model.forward(tensor)
            torch.cuda.synchronize()
            forward_times.append(time.perf_counter() - start)

            # 5. Postprocess
            start = time.perf_counter()
            mask = model.postprocess(output, meta)
            postprocess_times.append(time.perf_counter() - start)

            # 6. Save mask (no compression for speed)
            start = time.perf_counter()
            cv2.imwrite("temp_mask.png", mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            save_mask_times.append(time.perf_counter() - start)

            # 7. Create and save overlay (optimized version)
            start = time.perf_counter()
            # Optimized overlay creation (numpy direct assignment)
            h, w = img.shape[:2]
            overlay = np.empty((h, w, 4), dtype=np.uint8)
            overlay[:, :, :3] = img
            overlay[:, :, 3] = mask
            cv2.imwrite("temp_overlay.png", overlay, [cv2.IMWRITE_PNG_COMPRESSION, 0])
            save_overlay_times.append(time.perf_counter() - start)

    # Cleanup
    Path("temp_mask.png").unlink(missing_ok=True)
    Path("temp_overlay.png").unlink(missing_ok=True)

    # Results
    print("\n" + "="*60)
    print("FULL PIPELINE BREAKDOWN (per image)")
    print("="*60)

    def stats(times, name):
        times = times[20:]  # Skip warmup
        avg = sum(times) / len(times) * 1000
        print(f"{name:25s}: {avg:6.2f}ms")
        return avg

    load_avg = stats(load_times, "1. Load image (cv2.imread)")
    cvt_avg = stats(cvt_times, "2. BGR->RGB conversion")
    pre_avg = stats(preprocess_times, "3. Preprocess + transfer")
    fwd_avg = stats(forward_times, "4. Forward (TensorRT)")
    post_avg = stats(postprocess_times, "5. Postprocess")
    save_m_avg = stats(save_mask_times, "6. Save mask (cv2.imwrite)")
    save_o_avg = stats(save_overlay_times, "7. Create+save overlay")

    total = load_avg + cvt_avg + pre_avg + fwd_avg + post_avg + save_m_avg + save_o_avg
    print("-"*60)
    print(f"{'Total per image':25s}: {total:6.2f}ms")
    print(f"{'Throughput':25s}: {1000/total:.1f} images/sec")

    print("\n" + "="*60)
    print("BREAKDOWN BY CATEGORY")
    print("="*60)
    io_time = load_avg + save_m_avg + save_o_avg
    gpu_time = pre_avg + fwd_avg + post_avg
    cpu_time = cvt_avg
    print(f"I/O (load+save):     {io_time:6.2f}ms ({io_time/total*100:4.1f}%)")
    print(f"GPU (pre+fwd+post):  {gpu_time:6.2f}ms ({gpu_time/total*100:4.1f}%)")
    print(f"CPU (color convert): {cpu_time:6.2f}ms ({cpu_time/total*100:4.1f}%)")


if __name__ == "__main__":
    profile_full()
