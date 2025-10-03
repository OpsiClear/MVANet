# Multi-GPU Threading Overhead Analysis

## Current Bottlenecks

### 1. Thread Creation/Destruction Per Batch
**Current**: New threads created for every chunk (5 images)
```python
for i in range(0, len(files), chunk_size):  # Every 5 images
    chunk = files[i:i+5]
    # Create fresh threads
    workers = []
    for device in devices:
        worker = threading.Thread(...)
        worker.start()
    # Wait for completion
    work_queue.join()
    # Threads exit
```

**Problem**: Thread creation overhead (~10-50ms per batch)
- 40 images / 5 chunk_size = 8 batches
- 8 batches × ~20ms overhead = ~160ms wasted

**Solution**: Persistent worker pool (threads run for entire session)

### 2. Queue Synchronization Overhead
**Current**: Multiple synchronization points
```python
work_queue.put(item)        # Lock/unlock per item
work_queue.task_done()      # Lock/unlock per item
work_queue.join()           # Blocks until all done
result_queue.get()          # Lock/unlock per item
```

**Problem**: Queue operations have ~0.1-1ms overhead each
- 40 images × 4 operations × 0.5ms = ~80ms wasted

**Solution**:
- Batch queue operations where possible
- Use lockless structures for single GPU case
- Consider using multiprocessing.Queue (faster than threading.Queue)

### 3. Data Marshalling
**Current**: Every tensor goes through queue
```python
work_queue.put((img_path, image, img_tensor, metadata))  # Copy references
result_queue.put((img_path, image, results_dict, None))  # Copy references
```

**Problem**: Reference counting overhead for large objects
- Minimal impact (~1-2ms per batch) but adds up

**Solution**: Direct function calls for single GPU (no queuing)

## Optimization Strategy

### Option 1: Persistent Worker Pool (Best for Multi-GPU)
Keep workers alive across all batches:
```python
class MultiGPUInferenceEngine:
    def __init__(...):
        # Start workers once at initialization
        self.work_queue = Queue()
        self.result_queue = Queue()
        self.workers = []
        for device in devices:
            worker = threading.Thread(target=self._worker_loop, daemon=True)
            worker.start()
            self.workers.append(worker)

    def _worker_loop(self, device):
        """Persistent worker - runs entire session"""
        while True:
            item = self.work_queue.get()
            if item is "SHUTDOWN":
                break
            # Process item
            self.work_queue.task_done()

    def process_images(self, image_data):
        # Just add to existing queue - no thread creation
        for data in image_data:
            self.work_queue.put(data)
        self.work_queue.join()
```

**Benefit**: Eliminates thread creation overhead (~160ms saved)

### Option 2: Lockless Single GPU Fast Path
For 1 GPU, skip threading entirely:
```python
def process_images(self, image_data, use_tta):
    if self.num_gpus == 1:
        # Direct processing - no threading
        return [self._process_single(data, use_tta) for data in image_data]
    else:
        # Multi-GPU threaded path
        ...
```

**Benefit**: Eliminates all threading overhead for 1 GPU (~200-300ms saved)

### Option 3: Batch Processing
Process multiple images per GPU call:
```python
# Instead of: for each image, call model(tensor)
# Do: tensors_batch = stack([tensor1, tensor2, ...])
#     model(tensors_batch)  # Single call, batch processing
```

**Benefit**: Reduces GPU kernel launch overhead, better GPU utilization

## Recommended Approach

**For general multi-GPU**: Option 1 (Persistent workers)
- Eliminates thread creation overhead
- Works well with any number of GPUs
- Clean architecture

**For optimal 1 GPU performance**: Option 1 + Option 2 combined
- Persistent workers for 2+ GPUs
- Fast path for 1 GPU (no threading)
- Best of both worlds

## Expected Performance Gain

**Current**:
- Single GPU path: 7.62s (0.19s/img)
- Multi-GPU 1 GPU: 11.6s (0.29s/img)
- Overhead: +4.0s (53% slower)

**After Option 1** (persistent workers):
- Thread creation overhead: -160ms
- Expected: ~11.4s → ~10.2s

**After Option 1 + 2** (persistent + fast path):
- All threading overhead eliminated for 1 GPU
- Expected: ~11.4s → ~7.8s (matches single GPU path)

**After Option 1 + 3** (persistent + batching):
- For multi-GPU setups (2+ GPUs)
- GPU utilization improvement: +20-30%
- Expected: Near-linear scaling with GPU count
