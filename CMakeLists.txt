# Require CMake 3.18 or newer, including support for CUDA_ARCHITECTURES.
cmake_minimum_required(VERSION 3.18)

# Select the MPI wrapper before CMake enables and detects the C++ compiler.
# find_program locates mpicxx on the search path; REQUIRED stops configuration
# if it cannot be found. Load the intended MPI module before running CMake.
find_program(MPI_CXX_WRAPPER NAMES mpicxx REQUIRED)

# CACHE records the compiler path in the build directory's CMakeCache.txt.
# FORCE replaces an existing cache value. Use a fresh build directory when
# switching toolchains, since other compiler-detection results are cached too.
set(CMAKE_CXX_COMPILER "${MPI_CXX_WRAPPER}" CACHE FILEPATH "MPI C++ compiler" FORCE)

# Link the host CUDA runtime dynamically (libcudart.so), rather than embedding
# libcudart_static.a. This does not select MPI's GPU communication transport.
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

# Name the project and enable compiler detection for C++ and CUDA.
# CUDA sources use the CUDA compiler (normally nvcc); mpicxx handles C++.
project(cuda_aware_mpi_workshop LANGUAGES CXX CUDA)

# CUDA compilation uses nvcc, so obtain MPI settings explicitly even though
# the C++ compiler wrapper already provides them implicitly.
# Skip FindMPI's built-in-compiler shortcut and interrogate the chosen wrapper
# for MPI headers, libraries, and flags needed outside ordinary C++ compilation.
set(MPI_ASSUME_NO_BUILTIN_MPI TRUE)
set(MPI_CXX_COMPILER "${MPI_CXX_WRAPPER}")
# Require the C++ component and provide the imported MPI::MPI_CXX target.
find_package(MPI REQUIRED COMPONENTS CXX)

# Require C++14 for both C++ and CUDA compilation. REQUIRED prevents CMake
# from falling back to an older language standard if this one is unavailable.
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Cache the GPU architecture selection so it can be supplied at configuration
# time, for example -DCMAKE_CUDA_ARCHITECTURES=70 for Gadi's Volta GPUs.
# Without FORCE, an existing cache value is preserved.
set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING
    "CUDA architectures to compile for (70 is NVIDIA Volta)")

# CMAKE_BINARY_DIR is the build tree selected by cmake -B (usually build/).
# Put executables in its bin/ subdirectory, as expected by the PBS scripts.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Each name identifies both an executable target and its src/<name>.cu file.
# Add a new example here to include it in the build loop below.
set(WORKSHOP_TARGETS
    01-rank-device
    02-staged-pingpong
    03-device-pingpong
    04-jacobi-blocking
    05-jacobi-overlap
    06-bandwidth
    07-device-collectives
    08-strided-halo
    09-managed-memory
    10-preinit-device
    12-stream-aware-mpi
    13-warmup-run)

# Apply the same build settings to every example. A single target can be built
# with: cmake --build build --target 13-warmup-run
foreach(target IN LISTS WORKSHOP_TARGETS)

  # Create an executable from this example's CUDA source file.
  add_executable(${target} "src/${target}.cu")

  # Import MPI's usage requirements instead of hard-coding library paths.
  # PRIVATE applies this dependency to this target without exporting it.
  target_link_libraries(${target} PRIVATE MPI::MPI_CXX)

  # Make project headers such as common.h available during compilation.
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")

  # This generator expression adds -lineinfo only for CUDA compilation.
  # It records device-source line information useful for profiling/debugging.
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
    
  # Use the C++ compiler (mpicxx) for the final host link, even though the
  # target contains only .cu files. This keeps the MPI wrapper in the link path.
  # Separable compilation enables relocatable device code and a device-link
  # step, allowing device symbols to be resolved across CUDA translation units.
  # It is independent of the shared-versus-static CUDA runtime choice above.
  set_target_properties(${target} PROPERTIES
    LINKER_LANGUAGE CXX
    CUDA_SEPARABLE_COMPILATION ON)
endforeach()
