from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import pybind11

# Define the extension module
ext_modules = [
    Pybind11Extension(
        "gcocpp",
        [
            "bindings/ctswap_bindings.cpp",
            "src/planners/base_planner.cpp",
            "src/planners/ctswap.cpp",
            "src/planners/pibt.cpp",
            "src/planners/gspi.cpp",
            "src/planners/object_gspi.cpp",
            "src/planners/object_centric_a_star.cpp",
            "src/heuristics/euclidean_heuristic.cpp",
            "src/heuristics/bwd_dijkstra_heuristic.cpp",
            "src/robot.cpp", 
            "src/world.cpp",
            "src/obstacles/obstacles.cpp",
            "src/objects/objects.cpp",
            "src/geometric_shapes/geometric_shapes.cpp",
            "src/utils.cpp",
            "src/utils/progress_tracker.cpp"
        ],
        include_dirs=["include"],
        language='c++',
        cxx_std=17,
    ),
]

setup(
    name="gcocpp",
    version="0.1.0",
    author="Yorai Shaoul",
    author_email="yorai@cmu.edu",
    description="Python bindings for GCo C++ planners",
    long_description="",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "pybind11>=2.6.0",
    ],
) 