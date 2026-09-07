#ifndef CUDA_MPI_COMMON_H
#define CUDA_MPI_COMMON_H


#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mpi.h>


#define CUDA_CHECK(call)                                                       \
  do {                                                                         \
    cudaError_t e_ = (call);                                                   \
    if (e_ != cudaSuccess) {                                                   \
      std::fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,       \
                   cudaGetErrorString(e_));                                    \
      MPI_Abort(MPI_COMM_WORLD, 2);                                            \
    }                                                                          \
  } while (0)


#define MPI_CHECK(call)                                                        \
  do {                                                                         \
    int e_ = (call);                                                           \
    if (e_ != MPI_SUCCESS) {                                                   \
      char m_[MPI_MAX_ERROR_STRING];                                           \
      int n_;                                                                  \
      MPI_Error_string(e_, m_, &n_);                                           \
      std::fprintf(stderr, "MPI error %s:%d: %.*s\n", __FILE__, __LINE__, n_,  \
                   m_);                                                        \
      MPI_Abort(MPI_COMM_WORLD, e_);                                           \
    }                                                                          \
  } while (0)


inline int select_device(MPI_Comm world, int *local_rank_out = nullptr)
{
  MPI_Comm local;
  MPI_CHECK(MPI_Comm_split_type(world, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL,
                                &local));
  int local_rank, devices;
  MPI_CHECK(MPI_Comm_rank(local, &local_rank));
  CUDA_CHECK(cudaGetDeviceCount(&devices));


  if (!devices) {
    std::fprintf(stderr, "No visible CUDA devices\n");
    MPI_Abort(world, 3);
  }


  int device = local_rank % devices;
  CUDA_CHECK(cudaSetDevice(device));


  MPI_CHECK(MPI_Comm_free(&local));


  if (local_rank_out) {
    *local_rank_out = local_rank;
  }


  return device;
}


/* Select before MPI_Init for MPI libraries that initialise GPU state there. */
inline int preselect_device_from_env()
{
  const char *names[] = {"OMPI_COMM_WORLD_LOCAL_RANK", "MPI_LOCALRANKID",
                         "MV2_COMM_WORLD_LOCAL_RANK", "SLURM_LOCALID", nullptr};
  const char *value = nullptr;
  for (int i = 0; names[i] && !value; ++i) {
    value = std::getenv(names[i]);
  }
  if (!value)
    return -1;


  int devices = 0;
  cudaError_t error = cudaGetDeviceCount(&devices);
  if (error != cudaSuccess || devices == 0)
    return -1;


  int device = std::atoi(value) % devices;
  return cudaSetDevice(device) == cudaSuccess ? device : -1;
}
#endif
