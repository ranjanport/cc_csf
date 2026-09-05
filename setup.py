from setuptools import setup, Extension
import pybind11

sources = [
    "cc_csf.cpp",
    "src/c2cdist.cpp",
    "src/Cloth.cpp",
    "src/Constraint.cpp",
    "src/CSF.cpp",
    "src/Particle.cpp",
    "src/point_cloud.cpp",
    "src/Rasterization.cpp",
    "src/XYZReader.cpp",
]

ext_modules = [
    Extension(
        "cc_csf",
        sources=sources,
        include_dirs=[pybind11.get_include(), "src"],
        extra_compile_args=["-O3", "-march=native", "-ffast-math", "-fopenmp", "-std=c++14"],
        extra_link_args=["-fopenmp"],
        define_macros=[("CSF_USE_OPENMP", "1")],
    ),
]

setup(
    name="cc_csf",
    version="1.0.0",
    description="High-performance Zero-Copy OpenMP Pybind11 Cloth Simulation Filter (CSF)",
    ext_modules=ext_modules,
)
