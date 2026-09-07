#include "00-common.h"


__global__ void set_value(int *value, int x)
{
  *value = x;
}


int main(int argc, char **argv)
{
  MPI_CHECK(MPI_Init(&argc, &argv));
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);


  if (size != 2) {
    if (!rank)
      std::fprintf(stderr, "Run with exactly 2 ranks\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }


  int device = select_device(MPI_COMM_WORLD);


  int *value;
  CUDA_CHECK(cudaMallocManaged(&value, sizeof(int)));
  CUDA_CHECK(cudaMemAdvise(value, sizeof(int),
                           cudaMemAdviseSetPreferredLocation, device));


  set_value<<<1, 1>>>(value, rank);
  CUDA_CHECK(cudaDeviceSynchronize());


  MPI_CHECK(MPI_Sendrecv_replace(value, 1, MPI_INT, 1 - rank, 9, 1 - rank, 9,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE));


  CUDA_CHECK(cudaMemPrefetchAsync(value, sizeof(int), cudaCpuDeviceId));
  CUDA_CHECK(cudaDeviceSynchronize());


  int ok = *value == 1 - rank;


  std::printf("rank %d: managed-memory value %d (expected %d) %s\n", rank,
              *value, 1 - rank, ok ? "PASS" : "FAIL");


  CUDA_CHECK(cudaFree(value));


  MPI_Finalize();
  return ok ? 0 : 1;
}
