cmake_minimum_required(VERSION 3.10)

# Set project name and version
project(CTSWAP VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json for IntelliSense
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Option to build Python bindings (default OFF)
option(GCO_BUILD_PYTHON_BINDINGS "Build pybind11 Python bindings" OFF)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# ----------------------------------------------------------------------------------
# Core GCo C++ sources (shared by Python bindings and example executables)
# ----------------------------------------------------------------------------------

set(GCO_CORE_SOURCES
    src/robot.cpp
    src/world.cpp
    src/utils.cpp
    src/utils/progress_tracker.cpp
    src/geometric_shapes/geometric_shapes.cpp
    src/obstacles/obstacles.cpp
    src/objects/objects.cpp
)

# Planner sources
file(GLOB GCO_PLANNER_SOURCES src/planners/*.cpp)

# Heuristic sources
file(GLOB GCO_HEURISTIC_SOURCES src/heuristics/*.cpp)

# Aggregate
set(GCO_ALL_SOURCES
    ${GCO_CORE_SOURCES}
    ${GCO_PLANNER_SOURCES}
    ${GCO_HEURISTIC_SOURCES}
)

# Static library to reuse between executables and bindings
add_library(gco_static STATIC ${GCO_ALL_SOURCES})

# Public include directories
target_include_directories(gco_static PUBLIC
    ${CMAKE_SOURCE_DIR}/include
)


# ----------------------------------------------------------------------------------
# Find pybind11 (optional)
find_package(pybind11 CONFIG QUIET)
set(BUILD_PYTHON_BINDINGS ${pybind11_FOUND})

# Add examples subdirectory
add_subdirectory(examples)

# ----------------------------------------------------------------------------------
# Python bindings
# ----------------------------------------------------------------------------------

if(GCO_BUILD_PYTHON_BINDINGS)
    find_package(pybind11 CONFIG REQUIRED)
    pybind11_add_module(gcocpp 
        bindings/ctswap_bindings.cpp
        src/planners/base_planner.cpp
        src/planners/ctswap.cpp
        src/planners/pibt.cpp
        src/planners/gspi.cpp
        src/planners/object_gspi.cpp
        src/heuristics/euclidean_heuristic.cpp
        src/heuristics/bwd_dijkstra_heuristic.cpp
        src/robot.cpp
        src/world.cpp
        src/obstacles/obstacles.cpp
        src/objects/objects.cpp
        src/geometric_shapes/geometric_shapes.cpp
        src/utils.cpp
        src/utils/progress_tracker.cpp
    )
    target_include_directories(gcocpp PRIVATE 
        include
    )
endif()

# Set output directory for executables
# set_target_properties(ctswap_example PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
# )

# Optional: Add compiler warnings and debug flags
# if(MSVC)
#     target_compile_options(ctswap_example PRIVATE /W4)
#     if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#         target_compile_options(ctswap_example PRIVATE /Zi /Od)
#     endif()
# else()
#     target_compile_options(ctswap_example PRIVATE -Wall -Wextra -Wpedantic)
#     if(CMAKE_BUILD_TYPE STREQUAL "Debug")
#         target_compile_options(ctswap_example PRIVATE -g -O0)
#     endif()
# endif()