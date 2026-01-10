cmake_minimum_required(VERSION 3.16)

# Note: this file is within gco-cpp/examples directory. The parent CMakeLists sets project and gco_static.

# ----------------------------------------------------------------------------------
# Common settings
# ----------------------------------------------------------------------------------

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# ----------------------------------------------------------------------------------
# Common utility object library (built once)
# ----------------------------------------------------------------------------------
set(COMMON_SRC
    common/configuration_generators.cpp
    common/task_setup.cpp
    common/visualization_utils.cpp
)
add_library(gco_example_common OBJECT ${COMMON_SRC})
target_include_directories(gco_example_common PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_options(gco_example_common PRIVATE -Wall -Wextra -O2 -g)

# ----------------------------------------------------------------------------------
# Helper macro to create example executables
# ----------------------------------------------------------------------------------
function(gco_add_example target_name)
    add_executable(${target_name} ${ARGN})
    target_link_libraries(${target_name} PRIVATE gco_example_common gco_static)
    target_compile_options(${target_name} PRIVATE -Wall -Wextra -O2 -g)
    set_target_properties(${target_name} PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
endfunction()

# ----------------------------------------------------------------------------------
# Define examples
# ----------------------------------------------------------------------------------

# Planner Example
gco_add_example(planner_example planner_example.cpp)