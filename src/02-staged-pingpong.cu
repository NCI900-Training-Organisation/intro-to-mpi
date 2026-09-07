#include "00-common.h"

__global__ void fill(float *a, size_t n, float v)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) {
    a[i] = v;
  }
}

int main(int argc, char **argv)
{
  MPI_CHECK(MPI_Init(&argc, &argv));
  int rank, size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (size != 2) {
    if (!rank) {
      std::fprintf(stderr, "Run with exactly 2 ranks\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  select_device(MPI_COMM_WORLD);
  size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 1u << 20;
  float *d, *h;

  CUDA_CHECK(cudaMalloc(&d, n * sizeof(float)));
  CUDA_CHECK(cudaMallocHost(&h, n * sizeof(float)));
  fill<<<(n + 255) / 256, 256>>>(d, n, (float)rank);
  CUDA_CHECK(cudaMemcpy(h, d, n * sizeof(float), cudaMemcpyDeviceToHost));

  MPI_Barrier(MPI_COMM_WORLD);
  double t = MPI_Wtime();

  MPI_CHECK(MPI_Sendrecv_replace(h, (int)n, MPI_FLOAT, 1 - rank, 0, 1 - rank, 0,
                                 MPI_COMM_WORLD, MPI_STATUS_IGNORE));
  CUDA_CHECK(cudaMemcpy(d, h, n * sizeof(float), cudaMemcpyHostToDevice));
  t = MPI_Wtime() - t;
  float first;
  CUDA_CHECK(cudaMemcpy(&first, d, sizeof(float), cudaMemcpyDeviceToHost));

  if (!rank) {
    std::printf("staged: %.1f MiB/rank, %.3f ms, received %.0f\n",
                n * sizeof(float) / 1048576.0, t * 1e3, first);
  }

  CUDA_CHECK(cudaFreeHost(h));
  CUDA_CHECK(cudaFree(d));
  MPI_Finalize();
  return 0;
}
