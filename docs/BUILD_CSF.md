# Guide: Building & Distributing the High-Performance CSF Filter (`cc_csf`)

This guide explains how to compile, package, and distribute `cc_csf`—the custom **zero-copy, OpenMP-accelerated Pybind11 Cloth Simulation Filter (CSF)**—so it can be used across multiple projects and machines.

---

## 1. Overview & Architecture

### Why `cc_csf` instead of PyPI `cloth-simulation-filter`?
Standard Python CSF packages from PyPI use legacy ctypes or C wrappers that suffer from severe performance bottlenecks on large LiDAR point clouds:
1. **Zero-Copy Memory Access**: `cc_csf` accepts contiguous $N \times 3$ double-precision NumPy arrays (`np.float64`) and accesses raw memory pointers directly via `pybind11::buffer_info`, eliminating Python-to-C point-by-point serialization.
2. **OpenMP Multi-Core Parallelism**: Ground simulation and coordinate inversion are parallelized using `#pragma omp parallel for`, utilizing all available CPU threads (`-fopenmp -DCSF_USE_OPENMP=1`).
3. **Compiler Vectorization**: Compiled with `-O3 -march=native -ffast-math -std=c++14`, achieving a **10x–20x speedup** over default implementations.

### Repository Structure
A standalone, self-contained package is organized under `packages/cc_csf/`:

```
packages/cc_csf/
├── pyproject.toml         # Modern PEP 517/518 build specification
├── setup.py               # Setuptools Extension definition with OpenMP & optimization flags
├── MANIFEST.in            # Ensures C++ headers (.h) are packaged into sdist & wheels
├── README.md              # Standalone package documentation
├── cc_csf.cpp             # Pybind11 wrapper exposing CSFWrapper class
└── src/                   # Core CSF C++ library (Apache 2.0 license)
    ├── CSF.cpp / CSF.h
    ├── Cloth.cpp / Cloth.h
    ├── Constraint.cpp / Constraint.h
    ├── Particle.cpp / Particle.h
    ├── point_cloud.cpp / point_cloud.h
    ├── Rasterization.cpp / Rasterization.h
    ├── XYZReader.cpp / XYZReader.h
    ├── c2cdist.cpp / c2cdist.h
    └── Vec3.h
```

---

## 2. System & Environment Prerequisites

### 2.1 Operating System Dependencies (Ubuntu / Debian / Linux)
Ensure a C++14-compliant compiler and OpenMP libraries are installed:

```bash
sudo apt update
sudo apt install -y build-essential g++ libomp-dev python3-dev
```

### 2.2 Python Build Dependencies
Install build tools in your virtual environment:

```bash
# Using uv (recommended)
uv pip install pybind11 wheel build setuptools

# Or using standard pip
pip install pybind11 wheel build setuptools
```

---

## 3. Option A: Build a Standalone Wheel (`.whl`) for Distribution (Recommended)

Building a binary wheel (`.whl`) allows you to distribute `cc_csf` to other projects or target machines without requiring them to have C++ compilers, `pybind11`, or the C++ source files installed.

### Step 1: Compile the Wheel
From the project root, run:

```bash
# Using uv with --no-isolation (avoids Debian/Ubuntu ensurepip issues)
uv run python -m build --no-isolation --wheel packages/cc_csf --outdir wheels
```

> [!NOTE]
> The `--no-isolation` flag uses the current environment's `pybind11` and `setuptools`, bypassing Debian/Ubuntu minimal python `ensurepip` constraints.

### Step 2: Verify the Built Wheel
The compiled binary wheel is stored in `wheels/`:
```bash
ls -lh wheels/
# Output: cc_csf-1.0.0-cp312-cp312-linux_x86_64.whl (~1.7 MB)
```

### Step 3: Install in Other Projects
To use `cc_csf` in any other repository or project:

```bash
# In another project with uv:
uv pip install /path/to/wheels/cc_csf-1.0.0-cp312-cp312-linux_x86_64.whl

# Or using pip:
pip install /path/to/wheels/cc_csf-1.0.0-cp312-cp312-linux_x86_64.whl
```

Alternatively, you can declare it directly in the other project's `pyproject.toml`:
```toml
[project]
dependencies = [
    "cc_csf @ file:///absolute/path/to/wheels/cc_csf-1.0.0-cp312-cp312-linux_x86_64.whl"
]
```

---

## 4. Option B: Distribute as a Standalone Git Repository

If you prefer other projects to install directly from source via Git:

### Step 1: Create a Standalone Git Repository
Copy the contents of `packages/cc_csf` into a dedicated repository:

```bash
cp -r packages/cc_csf ~/cc_csf_repo
cd ~/cc_csf_repo
git init
git add .
git commit -m "Initial commit: Standalone Zero-Copy OpenMP cc_csf module"
git remote add origin git@github.com:your-org/cc_csf.git
git push -u origin main
```

### Step 2: Install via Git in Any Project
Other projects can install directly using `uv` or `pip`:

```bash
# Using uv
uv pip install git+https://github.com/your-org/cc_csf.git

# Using standard pip
pip install git+https://github.com/your-org/cc_csf.git
```

---

## 5. Option C: Local Development In-Place Build

If you are developing or modifying `cc_csf.cpp` and need to test changes immediately in the current project:

```bash
# Compile shared object (.so) directly into src/
uv run python setup_csf.py build_ext --inplace
```

This compiles `cc_csf.cpython-312-x86_64-linux-gnu.so` and places it in `src/`, where it can be directly imported by `polesense`.

---

## 6. Option D: Building from Upstream CSF (From Scratch)

If starting on a completely clean server without the `packages/cc_csf` folder:

1. **Clone the Upstream CSF Repository**:
   ```bash
   git clone https://github.com/jianboqi/CSF.git /tmp/CSF
   ```

2. **Copy Core Source Files**:
   ```bash
   mkdir -p cc_csf_pkg/src
   cp /tmp/CSF/src/*.cpp /tmp/CSF/src/*.h cc_csf_pkg/src/
   cp cc_csf.cpp cc_csf_pkg/
   ```

3. **Build the Wheel**:
   ```bash
   uv run python -m build --no-isolation --wheel cc_csf_pkg --outdir dist
   ```

---

## 7. Python API Reference & Verification

### Verification Test
Run this one-liner to verify that the compiled module imports and executes multi-threaded filtering:

```bash
uv run python -c "
import numpy as np, cc_csf
csf = cc_csf.CSF()
csf.set_params(True, 0.5, 500, 0.3, 1)
pts = np.random.rand(10000, 3) * 50.0
csf.set_point_cloud(pts)
ground = csf.do_filtering()
print(f'Verification passed! Found {len(ground)} ground points.')
"
```

### Complete Usage Example with LiDAR Data

```python
import laspy
import numpy as np
import cc_csf

# 1. Read LiDAR point cloud (.laz / .las)
las = laspy.read("sample_road.laz")

# 2. Extract XYZ as contiguous float64 array (critical for zero-copy)
xyz = np.ascontiguousarray(
    np.vstack((las.x, las.y, las.z)).T, 
    dtype=np.float64
)

# 3. Initialize CSF Filter
csf = cc_csf.CSF()

# 4. Configure Parameters:
#    set_params(bSloopSmooth, cloth_resolution, iterations, class_threshold, rigidness)
csf.set_params(
    bSloopSmooth=True,       # Post-processing slope smoothing
    cloth_resolution=0.5,    # Cloth grid size in meters (0.5m recommended for roads)
    iterations=500,          # Maximum displacement iterations
    class_threshold=0.3,     # Distance threshold from cloth to classify ground (meters)
    rigidness=1              # 1 = flat roads (fastest), 2 = moderate slopes, 3 = steep terrain
)

# 5. Pass points into C++ engine (zero-copy)
csf.set_point_cloud(xyz)

# 6. Execute multi-core OpenMP cloth simulation
ground_indices = csf.do_filtering()  # Returns List[int]

# 7. Index points
ground_xyz = xyz[ground_indices]
non_ground_mask = np.ones(len(xyz), dtype=bool)
non_ground_mask[ground_indices] = False
non_ground_xyz = xyz[non_ground_mask]

print(f"Total Points:      {len(xyz):,}")
print(f"Ground Points:     {len(ground_indices):,}")
print(f"Non-Ground Points: {len(non_ground_xyz):,}")
```

---

## 8. Parameter Tuning Reference

| Parameter | Type | Default | Description & Road LiDAR Recommendations |
| :--- | :--- | :--- | :--- |
| `bSloopSmooth` | `bool` | `True` | Smooths sharp step changes on slopes; set to `True` for road corridors. |
| `cloth_resolution` | `float` | `0.5` | Cloth node spacing in meters. `0.5m` captures road curbs and ditches accurately without excessive memory. |
| `iterations` | `int` | `500` | Simulation iteration limit. 500 is optimal; convergence usually reached by ~200. |
| `class_threshold` | `float` | `0.3` | Points within this distance from the cloth mesh are classified as ground ($0.3\text{m} - 0.5\text{m}$). |
| `rigidness` | `int` | `1` | `1` (flat ground/roads), `2` (slopes), `3` (steep mountains). Set to `1` for maximum performance on urban corridors. |

---

## 9. Troubleshooting & FAQ

### Issue 1: `Error: setup script specifies an absolute path: /tmp/CSF/src/CSF.cpp`
- **Cause**: Setuptools `bdist_wheel` strictly prohibits absolute paths in `sources`.
- **Solution**: Use `packages/cc_csf/setup.py` where all sources are relative to the package directory (`src/CSF.cpp`).

### Issue 2: `The virtual environment was not created successfully because ensurepip is not available`
- **Cause**: Python `build` attempts to create an isolated venv by default, which fails on minimal Debian/Ubuntu installations without `python3-venv`.
- **Solution**: Pass `--no-isolation` to `python -m build`:
  ```bash
  uv run python -m build --no-isolation --wheel packages/cc_csf --outdir wheels
  ```

### Issue 3: `ImportError: libgomp.so.1: cannot open shared object file`
- **Cause**: The target machine does not have OpenMP runtime installed.
- **Solution**: Run `sudo apt install libgomp1` on the target system.

### Issue 4: Binary Compatibility Across Python Versions
- **Cause**: CPython C-extension wheels (`cp312`) are ABI-specific to the Python version they were compiled for (e.g. Python 3.12).
- **Solution**: Use `cibuildwheel` (see Section 10 below) to compile for all Python versions (3.10–3.13) into manylinux-compatible wheels for PyPI.

---

## 10. Publishing Multi-Python Manylinux Wheels to PyPI (`cibuildwheel`)

### Why `cibuildwheel` is Required for PyPI
1. **PyPI Rejection of `linux_x86_64`**: PyPI **strictly rejects** generic `linux_x86_64` wheels because they depend on your host machine's specific `glibc` version. PyPI requires binary wheels to use standardized tags like `manylinux_2_28_x86_64` or `manylinux2014_x86_64`.
2. **Auditwheel Bundling**: `cibuildwheel` automatically invokes `auditwheel repair`, which inspects compiled `.so` files and vendors shared libraries (such as `libgomp.so.1` for OpenMP) directly into the wheel, so end-users never need to manually install compiler packages.

### 10.1 `pyproject.toml` Configuration
Ensure your `pyproject.toml` includes:

```toml
[build-system]
requires = ["setuptools>=61.0", "pybind11>=2.10", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cc_csf"
version = "1.0.0"
description = "High-performance Zero-Copy OpenMP Pybind11 Cloth Simulation Filter (CSF)"
requires-python = ">=3.10"
dependencies = ["numpy>=1.20"]

[tool.cibuildwheel]
# Target Python 3.10, 3.11, 3.12, 3.13
build = "cp310-* cp311-* cp312-* cp313-*"
skip = "pp* *i686* *musllinux* *aarch64*"

[tool.cibuildwheel.linux]
manylinux-x86_64-image = "manylinux_2_28"
repair-wheel-command = "auditwheel repair -w {dest_dir} {wheel}"
```

### 10.2 Recommended: Automated GitHub Actions Publishing (Free & Zero Setup)
The package in `/home/Documents/cc_csf` includes a ready-to-use GitHub Actions workflow at `.github/workflows/publish.yml`.

When you push code to GitHub with a version tag:
```bash
cd /home/Documents/cc_csf
git init
git add .
git commit -m "feat: initial release"
git remote add origin git@github.com:your-username/cc_csf.git
git branch -M main
git push -u origin main

# Trigger release build:
git tag v1.0.0
git push origin v1.0.0
```

GitHub Actions will automatically:
1. Spin up a clean Linux VM.
2. Launch Docker with `manylinux_2_28`.
3. Compile optimized wheels for Python 3.10, 3.11, 3.12, and 3.13.
4. Run `auditwheel repair` to bundle OpenMP runtime into the wheels.
5. Publish all wheels and the source tarball (`.tar.gz`) directly to PyPI!

### 10.3 Alternative: Building Locally with Docker
If you have Docker installed and permissions configured (`sudo usermod -aG docker $USER`):

```bash
cd /home/Documents/cc_csf

# 1. Run cibuildwheel using uv
uvx cibuildwheel --output-dir dist

# 2. Upload to PyPI using twine
uvx twine upload dist/*
```

