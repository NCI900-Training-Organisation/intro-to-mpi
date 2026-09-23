#include "common.h"


int main(int argc, char **argv)
{
  MPI_CHECK(MPI_Init(&argc, &argv));
  int rank, size, local_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);


  // Each MPI rank must pick a different CUDA device on the same node.
  // The shared-memory communicator groups ranks that are on the same host,
  // so their local rank values are 0, 1, 2, ... in order. The mapping is:
  //   device = local_rank % device_count
  // With a 2-rank, 2-GPU job, rank 0 gets local rank 0 and rank 1 gets local
  // rank 1, so the first process uses GPU 0 and the second uses GPU 1.
  int device = select_device(MPI_COMM_WORLD, &local_rank);
  cudaDeviceProp prop;
  CUDA_CHECK(cudaGetDeviceProperties(&prop, device));


  char host[MPI_MAX_PROCESSOR_NAME];
  int len;
  MPI_Get_processor_name(host, &len);


  for (int r = 0; r < size; r++) {
    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == r) {
      std::printf("rank %d/%d on %s: local rank %d -> GPU %d (%s)\n", rank,
                  size, host, local_rank, device, prop.name);
    }
  }


  MPI_CHECK(MPI_Finalize());
  return 0;
}
