cmake_minimum_required(VERSION 3.18)

project(cuda_aware_mpi_workshop LANGUAGES CXX CUDA)

find_package(MPI REQUIRED COMPONENTS CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_ARCHITECTURES 70 CACHE STRING
    "CUDA architectures to compile for (70 is NVIDIA Volta)")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

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
    12-stream-aware-mpi)

foreach(target IN LISTS WORKSHOP_TARGETS)
  add_executable(${target} "src/${target}.cu")
  target_link_libraries(${target} PRIVATE MPI::MPI_CXX)
  target_include_directories(${target} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src")
  target_compile_options(${target} PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
  set_target_properties(${target} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON)
endforeach()
