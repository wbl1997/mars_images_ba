cmake_minimum_required(VERSION 3.10)

# Project Name
project(MARS_Images_BA)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)

# Add Source Files
file(GLOB_RECURSE SOURCES src/*.cpp src/*.h)

# Include Directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add Executable
add_executable(MARS_Images_BA ${SOURCES})

# Link Libraries
# target_link_libraries(MARS_Images_BA <library_name>)

# Set Output Directories
set_target_properties(MARS_Images_BA PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin
)