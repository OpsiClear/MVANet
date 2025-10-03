# Template Plugin for Inference Engine

This is a template plugin for creating your own segmentation model plugin.

## How to Use This Template

1. **Copy this directory**:
   ```bash
   cp -r plugins/template plugins/your-model-name
   cd plugins/your-model-name
   ```

2. **Rename package**:
   - Rename `src/inference_engine_template/` to `src/inference_engine_yourmodel/`
   - Update all imports to use your package name

3. **Update pyproject.toml**:
   ```toml
   [project]
   name = "inference-engine-yourmodel"
   description = "Your model description"
   dependencies = [
       "inference-engine>=0.1.0",
       "torch>=2.8.0",
       # Your dependencies
   ]

   [project.entry-points."inference_engine.models"]
   yourmodel = "inference_engine_yourmodel:YourModelClass"
   ```

4. **Implement your model** in `model.py`:
   - Replace `TemplateModel` class name
   - Implement model loading in `load()`
   - Customize preprocessing/postprocessing
   - Update metadata

5. **Add your model architecture**:
   - Add your model files to the package
   - Import in `model.py`: `from .your_architecture import YourModel`

6. **Test**:
   ```bash
   pip install -e .
   inference-engine list  # Should show your model
   inference-engine infer --model yourmodel --model-path weights.pth --input-folder test/
   ```

## Required Methods

Your model class MUST implement:

- `load(model_path, device)` - Load weights
- `preprocess(image)` - RGB numpy → tensor + metadata
- `forward(tensor)` - Run inference
- `postprocess(output, metadata)` - Tensor → uint8 numpy mask
- `name` property - Model name
- `supports_tta` property - TTA support flag

## Image Formats

- **Input**: RGB numpy array (H, W, 3) from OpenCV
- **Output**: Grayscale mask (H, W) uint8 (0-255)

## Examples

See `plugins/mvanet/` for a complete working example.

## Documentation

See [PLUGIN_GUIDE.md](../../inference-engine/PLUGIN_GUIDE.md) for detailed instructions.
