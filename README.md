# ZebraXenoSeg

A segmentation tool for brain tumors in zebrafish xenografts. The program
loads the included TorchScript model and automatically uses CUDA when available,
otherwise falling back to CPU.

## Requirements

- Windows, macOS, or Linux
- Python 3.10 or later, 64-bit (required)
- For GPU inference: a supported NVIDIA GPU and up-to-date NVIDIA driver
- The included `ZebraXenoSeg_ResNet_v1.0.pt` TorchScript model

## Set up the environment

The PyTorch 2.8.0 CPU index does not provide a compatible wheel for every
Python version or architecture. If pip reports that the newest available
version is 2.4.1, the `py` launcher may be selecting an older Python or a
32-bit installation. Check the interpreter before creating the environment:

```powershell
py -3.10 --version
py -3.10 -c "import struct; print(struct.calcsize('P') * 8, 'bit')"
```

Use 64-bit Python 3.10 or later. Then, from the repository folder, create and
activate a virtual environment:

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\activate
py -m pip install --upgrade pip
```

Install PyTorch first, choosing one option:

**CPU:**

```powershell
py -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cpu
```

**NVIDIA GPU (CUDA 12.6 build):**

```powershell
py -m pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

Then install the remaining packages and check the installation:

```powershell
py -m pip install -r requirements.txt
py -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The requirements file pins the PyTorch version used to validate this model;
the CPU and CUDA commands select the corresponding PyTorch build. CUDA support
is optional—the script uses CPU if CUDA is unavailable.

## Run on the sample data

Run from the repository folder, with the virtual environment activated:

```powershell
py ZebraXenoSeg.py .\SampleData
```

The script processes TIFF files directly inside the specified folder (it does
not search subfolders). If no folder is specified, it defaults to
`./SampleData/`:

```powershell
py ZebraXenoSeg.py
```

For each TIFF image, results are written to `SampleData/Segmented/`:

- `<image-name>.png` — segmentation mask
- `<image-name>_overlay.png` — source image with the predicted region outlined
  in green

To process another folder, pass its path instead of `SampleData`, for example:

```powershell
py ZebraXenoSeg.py "C:\data\experiment1"
```
