# AVTR-1 on ONNX Runtime

[AVTR-1](https://github.com/avaturn-live/avtr-1) (audio-driven talking-head avatars) running with
every model as ONNX. No TensorRT engines, no CUDA plugin compilation.

`avtr1-onnx.ipynb` takes a blank Kaggle GPU session to a rendered video, then packages the
artifacts for reuse.

## What the notebook does

| Section | Cells | |
|---|---|---|
| 1. Setup | 0–5 | Python 3.12 venv via `uv`, deps, clone, make the `import tensorrt` lazy |
| 2. Runtime patches | 6–9 | Writes `warp_hybrid.py`, `onnx_only.py`, `build_onnx.py` |
| 3. Weights | 10–11 | Downloads ~3.4 GB from HF, exports the motion model to ONNX |
| 4. Verify | 12–15 | NaN check + rotation sweep, run twice; prints `VERIFY: PASS` |
| 5. Render | 16–24 | Renders to mp4, extracts frames, plays it inline |
| 6. Benchmark | 25–26 | Per-stage timings and real-time factor |
| 7. Production bundle | 27–30 | Copies code + weights to `/kaggle/working/prod/`, zips them, gives download links |
| 8. Publish | 31–36 | Stages the same artifacts and pushes them to a private HF repo |

Cells 18–19 (`check_hf.py`) diagnose Hugging Face access if the gated download 403s.

## Requirements

- NVIDIA GPU, CUDA 12
- Python 3.12+
- Hugging Face token with access to the gated [`avaturn-live/avtr-1`](https://huggingface.co/avaturn-live/avtr-1) repo

## Running it (Kaggle)

1. Enable GPU and Internet.
2. Add your HF token as a Kaggle secret named `HF_TOKEN`. For section 8, add a **write** token as
   `HF_WRITE_TOKEN`. Don't paste tokens into cells — the notebook is saved with its source.
3. Run all cells.

`onnxruntime-gpu` is pinned to `1.22.0`. Newer wheels are CUDA-13 builds and, against CUDA-12
torch, fail to load the CUDA provider and silently fall back to CPU.

## Output

- `/kaggle/working/demo_onnx.mp4` — the render
- `/kaggle/working/zips/` — `code.zip`, `onnx.zip`, `models.zip`, `avatars.zip`
- A private HF repo mirroring `$AVTR1_LOCAL_STORAGE/main/`, so later runs skip the download and
  the export entirely

## Running it outside Kaggle

The five scripts drop into an `avtr-1` checkout:

```bash
git clone https://github.com/avaturn-live/avtr-1.git && cd avtr-1
cp /path/to/this/repo/*.py .

python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pip install --no-deps -e .

export HF_TOKEN=hf_...
export AVTR1_LOCAL_STORAGE=./artifacts

python patch_no_trt.py .                          # once
python scripts/download_artifacts.py --workers 4  # once, ~3.4 GB
python build_onnx.py                              # once, ~2 min
python render.py --speech audio.wav --avatar maria --bg plain_white --out out.mp4
```

`--avatar` is any filename stem in `avatars_artifacts/reference_frames/`, `--bg` any id in
`avatars_artifacts/backgrounds/`. Add `--listen other.wav` for the dual-stream
(active-listening) mode, `--duration N` to trim.

| File | Runs | Purpose |
|---|---|---|
| `patch_no_trt.py` | once | Makes the module-level `import tensorrt` lazy |
| `build_onnx.py` | once | Exports the motion model to ONNX and converts it to float32 |
| `onnx_only.py` | every run | Patches the engine loaders to use ONNX Runtime |
| `warp_hybrid.py` | every run | Runs the warp network's 5-D `GridSample` ops in torch |
| `render.py` | every run | Render CLI |

## What needed fixing

1. `runtime/loader.py` imports TensorRT at module level, so the package won't import without it.
2. `warp_network.onnx` requires the TensorRT `GridSample3D` plugin. The plugin-free
   `warp_network_ori.onnx` uses standard 5-D `GridSample`, which ONNX Runtime cannot run on CUDA
   (no kernel) or CPU (4-D only) — so that graph is split and the op runs in torch.
3. TorchScript exports float constants as float64. TensorRT's parser demotes them, ONNX Runtime
   does not, and inputs are bound by the session's dtype without a check — producing NaN frames.
4. ONNX Runtime given torch's CUDA stream runs asynchronously and races with CPU-fallback nodes,
   producing intermittently blank frames.

Plus fixed batch sizes and symbolic output shapes, which TensorRT handled via optimization profiles.

## Performance

~2.2 s per 5-frame chunk on a T4 (≈0.09× real-time), against 84–232 ms/chunk for the TensorRT
build on Ampere+. Suitable for offline rendering, not live streaming. Section 6 reports per-stage
timings on your own GPU.

## Licence

These scripts are MIT. The model weights are not: `avaturn-live/avtr-1` is under the AVTR-1
Community License (non-commercial by default), and the bundled InsightFace detector and landmark
models are non-commercial research only.
