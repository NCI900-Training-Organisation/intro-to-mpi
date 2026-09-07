#include "00-common.h"

int main(int argc, char **argv)
{
  MPI_CHECK(MPI_Init(&argc, &argv));
  int rank, size, local_rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
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
