#include "00-common.h"

__global__ void set_value(double *value, double x)
{
  *value = x;
}

int main(int argc, char **argv)
{
  MPI_CHECK(MPI_Init(&argc, &argv));
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  select_device(MPI_COMM_WORLD);
  double *send, *sum;
  CUDA_CHECK(cudaMalloc(&send, sizeof(double)));
  CUDA_CHECK(cudaMalloc(&sum, sizeof(double)));
  set_value<<<1, 1>>>(send, rank + 1.0);
  CUDA_CHECK(cudaDeviceSynchronize());
  MPI_CHECK(MPI_Allreduce(send, sum, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
  double result;
  CUDA_CHECK(cudaMemcpy(&result, sum, sizeof(result), cudaMemcpyDeviceToHost));
  double expected = size * (size + 1.0) / 2.0;
  std::printf("rank %d: device Allreduce %.1f (expected %.1f) %s\n", rank,
              result, expected, result == expected ? "PASS" : "FAIL");
  CUDA_CHECK(cudaFree(send));
  CUDA_CHECK(cudaFree(sum));
  MPI_Finalize();
  return result == expected ? 0 : 1;
}
