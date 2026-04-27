cmake_minimum_required(VERSION 3.10)
project(HNSW_Local_Rewiring VERSION 1.0.0 LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Set build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0 -DDEBUG")
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
    set(CMAKE_CXX_FLAGS_DEBUG "/Od /DDEBUG")
endif()

# Add compile options
target_compile_options(hnsw_demo PRIVATE
    $<$<CONFIG:Release>:-O3 -DNDEBUG>
    $<$<CONFIG:Debug>:-g -O0 -DDEBUG>
    -Wall
    -Wextra
    -Wpedantic
)

# Check for filesystem library
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    target_link_libraries(hnsw_demo PRIVATE stdc++fs)
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS "9.0")
    target_link_libraries(hnsw_demo PRIVATE c++fs)
endif()

# Add executable
add_executable(hnsw_demo cpp_hnsw_demo.cpp)

# Set output directory
set_target_properties(hnsw_demo PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}
    RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}
)

# Installation
install(TARGETS hnsw_demo
    RUNTIME DESTINATION bin
    COMPONENT Runtime
)

# Custom targets for convenience
add_custom_target(run
    COMMAND hnsw_demo --help
    DEPENDS hnsw_demo
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running HNSW demo with help"
)

add_custom_target(test
    COMMAND hnsw_demo --dataset ./amazon --dimension 384 --ef-construction 200
    DEPENDS hnsw_demo
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
    COMMENT "Running HNSW demo with test parameters"
)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "Install prefix: ${CMAKE_INSTALL_PREFIX}")
