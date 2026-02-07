"""Profile PDFNet pipeline to identify bottlenecks.

Breaks down timing for each stage:
1. Preprocessing (CPU)
2. MoGe depth estimation (GPU)
3. PDFNet segmentation (GPU)
4. Postprocessing (CPU)
"""

import time
import logging
import torch
import torch.nn.functional as F
import numpy as np
from contextlib import contextmanager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Timer:
    """Context manager for timing code blocks."""

    def __init__(self, name: str, sync_cuda: bool = True):
        self.name = name
        self.sync_cuda = sync_cuda
        self.elapsed_ms = 0

    def __enter__(self):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.sync_cuda and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed_ms = (time.perf_counter() - self.start) * 1000


def profile_pipeline(model, test_image: np.ndarray, num_iterations: int = 10) -> dict:
    """Profile each stage of the pipeline."""

    timings = {
        "preprocess": [],
        "to_device": [],
        "denormalize": [],
        "resize_for_moge": [],
        "moge_inference": [],
        "depth_postprocess": [],
        "pdfnet_inference": [],
        "postprocess": [],
        "total": [],
    }

    device = model.device

    for i in range(num_iterations):
        # Total timing
        with Timer("total") as t_total:

            # 1. Preprocessing (CPU)
            with Timer("preprocess", sync_cuda=False) as t_preprocess:
                original_size = (test_image.shape[1], test_image.shape[0])
                resized = np.ascontiguousarray(
                    np.resize(test_image, (model.image_size[1], model.image_size[0], 3))
                )
                # Actually use cv2.resize for accurate timing
                import cv2
                resized = cv2.resize(test_image, model.image_size, interpolation=cv2.INTER_LINEAR)
                img_float = resized.astype(np.float32) / 255.0
                img_normalized = (img_float - model.mean) / model.std
                tensor = torch.from_numpy(img_normalized.transpose(2, 0, 1)).unsqueeze(0)

            # 2. Transfer to GPU
            with Timer("to_device") as t_to_device:
                tensor = tensor.to(device, non_blocking=False)

            # Now profile the forward pass components
            with torch.no_grad():
                H, W = tensor.shape[2:]

                # 3. Denormalize for MoGe
                with Timer("denormalize") as t_denorm:
                    mean = torch.tensor(model.mean, device=device, dtype=tensor.dtype).view(1, 3, 1, 1)
                    std = torch.tensor(model.std, device=device, dtype=tensor.dtype).view(1, 3, 1, 1)
                    rgb_tensor = tensor * std + mean

                # 4. Resize for MoGe (1024 -> 518)
                with Timer("resize_for_moge") as t_resize:
                    rgb_resized = F.interpolate(
                        rgb_tensor,
                        size=(model.depth_input_size, model.depth_input_size),
                        mode='bilinear',
                        align_corners=False
                    )

                # 5. MoGe inference
                with Timer("moge_inference") as t_moge:
                    if model.moge_trt_engine is not None:
                        normal_tensor = model.moge_trt_engine(rgb_resized)
                        depth_tensor = normal_tensor[..., 2]
                        depth_tensor = (depth_tensor + 1.0) / 2.0
                        depth_tensor = depth_tensor.unsqueeze(1)
                    else:
                        moge_output = model.moge_model.infer(rgb_resized)
                        depth_tensor = moge_output["depth"]
                        normal_tensor = moge_output["normal"]
                        if depth_tensor.dim() == 3:
                            depth_tensor = depth_tensor.unsqueeze(1)

                # 6. Depth postprocessing (upsample + repeat)
                with Timer("depth_postprocess") as t_depth_post:
                    depth_for_pdfnet = F.interpolate(
                        depth_tensor,
                        size=(H, W),
                        mode='bilinear',
                        align_corners=False
                    )
                    depth_for_pdfnet = depth_for_pdfnet.repeat(1, 3, 1, 1)

                # 7. PDFNet inference
                with Timer("pdfnet_inference") as t_pdfnet:
                    if model.trt_engine is not None:
                        mask_pred = model.trt_engine(tensor, depth_for_pdfnet).sigmoid()
                    else:
                        mask_pred, _ = model.pdfnet_model.inference(tensor, depth_for_pdfnet)

                # 8. Postprocessing (resize + to numpy)
                with Timer("postprocess") as t_postprocess:
                    mask_resized = F.interpolate(
                        mask_pred,
                        size=(original_size[1], original_size[0]),
                        mode="bilinear",
                        align_corners=False
                    )
                    mask_np = (mask_resized.squeeze() * 256).clamp(0, 255).cpu().numpy().astype(np.uint8)

        # Record timings
        timings["preprocess"].append(t_preprocess.elapsed_ms)
        timings["to_device"].append(t_to_device.elapsed_ms)
        timings["denormalize"].append(t_denorm.elapsed_ms)
        timings["resize_for_moge"].append(t_resize.elapsed_ms)
        timings["moge_inference"].append(t_moge.elapsed_ms)
        timings["depth_postprocess"].append(t_depth_post.elapsed_ms)
        timings["pdfnet_inference"].append(t_pdfnet.elapsed_ms)
        timings["postprocess"].append(t_postprocess.elapsed_ms)
        timings["total"].append(t_total.elapsed_ms)

    # Calculate statistics (skip first 2 iterations for warmup)
    results = {}
    for name, times in timings.items():
        times = times[2:]  # Skip warmup
        results[name] = {
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
        }

    return results


def print_results(results: dict, title: str):
    """Print profiling results as a table."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")

    total_mean = results["total"]["mean"]

    # Order of stages
    stages = [
        ("preprocess", "1. Preprocess (CPU)"),
        ("to_device", "2. Transfer to GPU"),
        ("denormalize", "3. Denormalize"),
        ("resize_for_moge", "4. Resize for MoGe"),
        ("moge_inference", "5. MoGe Inference"),
        ("depth_postprocess", "6. Depth Postprocess"),
        ("pdfnet_inference", "7. PDFNet Inference"),
        ("postprocess", "8. Postprocess (CPU)"),
    ]

    print(f"\n{'Stage':<30} {'Mean (ms)':<12} {'Std':<10} {'% of Total':<12}")
    print("-" * 70)

    accounted = 0
    for key, label in stages:
        r = results[key]
        pct = (r["mean"] / total_mean) * 100
        accounted += r["mean"]
        bar = "#" * int(pct / 2)
        print(f"{label:<30} {r['mean']:<12.2f} {r['std']:<10.2f} {pct:<6.1f}% {bar}")

    print("-" * 70)
    print(f"{'Total':<30} {total_mean:<12.2f} {results['total']['std']:<10.2f} {'100.0%':<12}")

    # Summary
    overhead = total_mean - accounted
    print(f"\n{'Measurement overhead:':<30} {overhead:.2f} ms")

    # Identify bottleneck
    sorted_stages = sorted(stages, key=lambda x: results[x[0]]["mean"], reverse=True)
    print(f"\n{'Top 3 bottlenecks:'}")
    for i, (key, label) in enumerate(sorted_stages[:3], 1):
        r = results[key]
        pct = (r["mean"] / total_mean) * 100
        print(f"  {i}. {label}: {r['mean']:.1f} ms ({pct:.1f}%)")


def main():
    device = torch.device("cuda:0")

    logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Create test image
    test_image = np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)

    from oc_masker_pdfnet import PDFNetModel

    # Test PyTorch configuration
    logger.info("\n" + "="*70)
    logger.info("Profiling PyTorch configuration...")
    logger.info("="*70)

    model_pytorch = PDFNetModel(use_tensorrt=False, use_moge_tensorrt=False)
    model_pytorch.load(None, device)
    model_pytorch.optimize_for_inference(device)

    # Warmup
    tensor, _ = model_pytorch.preprocess(test_image)
    for _ in range(3):
        with torch.no_grad():
            _ = model_pytorch.forward(tensor)

    results_pytorch = profile_pipeline(model_pytorch, test_image, num_iterations=15)
    print_results(results_pytorch, "PyTorch (no TensorRT)")

    del model_pytorch
    torch.cuda.empty_cache()

    # Test TensorRT configuration
    logger.info("\n" + "="*70)
    logger.info("Profiling TensorRT configuration...")
    logger.info("="*70)

    model_trt = PDFNetModel(use_tensorrt=True, use_moge_tensorrt=True)
    model_trt.load(None, device)
    model_trt.optimize_for_inference(device)

    # Warmup
    tensor, _ = model_trt.preprocess(test_image)
    for _ in range(3):
        with torch.no_grad():
            _ = model_trt.forward(tensor)

    results_trt = profile_pipeline(model_trt, test_image, num_iterations=15)
    print_results(results_trt, "TensorRT (dual)")

    # Comparison
    print(f"\n{'='*70}")
    print("  COMPARISON: PyTorch vs TensorRT")
    print(f"{'='*70}")

    stages = [
        ("moge_inference", "MoGe Inference"),
        ("pdfnet_inference", "PDFNet Inference"),
        ("total", "Total"),
    ]

    print(f"\n{'Stage':<25} {'PyTorch (ms)':<15} {'TensorRT (ms)':<15} {'Speedup':<10}")
    print("-" * 70)

    for key, label in stages:
        pt = results_pytorch[key]["mean"]
        trt = results_trt[key]["mean"]
        speedup = pt / trt
        print(f"{label:<25} {pt:<15.1f} {trt:<15.1f} {speedup:.2f}x")


if __name__ == "__main__":
    main()
