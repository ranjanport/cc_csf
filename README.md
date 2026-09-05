# cc_csf (Zero-Copy OpenMP Cloth Simulation Filter)

High-performance Python wrapper for the Cloth Simulation Filter (CSF) algorithm using Pybind11, OpenMP parallelization, and zero-copy NumPy array buffers.

## Installation

### From Pre-built Wheel
```bash
uv pip install cc_csf-1.0.0-cp312-cp312-linux_x86_64.whl
# or
pip install cc_csf-1.0.0-cp312-cp312-linux_x86_64.whl
```

### From Source (In-place or install)
```bash
uv pip install -e .
# or
pip install .
```

## Quick Start Example

```python
import numpy as np
import cc_csf

# Initialize filter
csf = cc_csf.CSF()

# Configure parameters:
# set_params(bSloopSmooth, cloth_resolution, iterations, class_threshold, rigidness)
csf.set_params(True, 0.5, 500, 0.3, 1)

# Pass Nx3 float64 numpy array (zero-copy buffer)
xyz = np.random.rand(100000, 3) * 50.0
csf.set_point_cloud(xyz)

# Execute OpenMP-accelerated filtering
# Returns list of integer indices belonging to ground points
ground_indices = csf.do_filtering()
print(f"Detected {len(ground_indices)} ground points out of {len(xyz)}")
```

# Building

## Operating System Dependencies (Ubuntu / Debian / Linux)
Ensure a C++14-compliant compiler and OpenMP libraries are installed:

```bash
sudo apt update
sudo apt install -y build-essential g++ libomp-dev python3-dev
```

## Python Build Dependencies
Install build tools in your virtual environment:

```bash
# Using uv (recommended)
uv pip install pybind11 wheel build setuptools

# Or using standard pip
pip install pybind11 wheel build setuptools
```

## Step 1: Compile the Wheel
From the project root, run:

```bash
# Using uv with --no-isolation (avoids Debian/Ubuntu ensurepip issues)
uv run python -m build --no-isolation --wheel cc_csf --outdir wheels
```
> [!NOTE]
> The `--no-isolation` flag uses the current environment's `pybind11` and `setuptools`, bypassing Debian/Ubuntu minimal python `ensurepip` constraints.

### Step 2: Verify the Built Wheel
The compiled binary wheel is stored in `wheels/`:
```bash
ls -lh wheels/
# Output: cc_csf-1.0.0-cp312-cp312-linux_x86_64.whl 
```
---


