#include "00-common.h"

__global__ void fill_bytes(unsigned char *p, size_t n, unsigned char value)
{
  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    p[i] = value;
}

static double exchange(void *buffer, int bytes, int peer, int warmup,
                       int iterations)
{
  for (int i = 0; i < warmup; ++i)
    MPI_CHECK(MPI_Sendrecv_replace(buffer, bytes, MPI_BYTE, peer, 7, peer, 7,
                                   MPI_COMM_WORLD, MPI_STATUS_IGNORE));
  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  double start = MPI_Wtime();
  for (int i = 0; i < iterations; ++i)
    MPI_CHECK(MPI_Sendrecv_replace(buffer, bytes, MPI_BYTE, peer, 7, peer, 7,
                                   MPI_COMM_WORLD, MPI_STATUS_IGNORE));
  return (MPI_Wtime() - start) / iterations;
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
  select_device(MPI_COMM_WORLD);
  const int max_bytes = 64 * 1024 * 1024, warmup = 10, iterations = 50;
  unsigned char *device, *host;
  CUDA_CHECK(cudaMalloc(&device, max_bytes));
  CUDA_CHECK(cudaMallocHost(&host, max_bytes));
  fill_bytes<<<(max_bytes + 255) / 256, 256>>>(device, max_bytes, rank);
  CUDA_CHECK(cudaDeviceSynchronize());
  if (!rank)
    std::printf("# bytes  host_MPI_us  device_MPI_us  device_GB/s\n");
  for (int bytes = 1; bytes <= max_bytes; bytes *= 2) {
    CUDA_CHECK(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
    double staged = exchange(host, bytes, 1 - rank, warmup, iterations);
    CUDA_CHECK(cudaDeviceSynchronize());
    double direct = exchange(device, bytes, 1 - rank, warmup, iterations);
    double worst[2], local[2] = {staged, direct};
    MPI_CHECK(
        MPI_Reduce(local, worst, 2, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD));
    if (!rank)
      std::printf("%8d %12.3f %14.3f %12.3f\n", bytes, worst[0] * 1e6,
                  worst[1] * 1e6, bytes / worst[1] / 1e9);
  }
  CUDA_CHECK(cudaFreeHost(host));
  CUDA_CHECK(cudaFree(device));
  MPI_Finalize();
  return 0;
}
