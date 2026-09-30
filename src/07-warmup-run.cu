#include "common.h"
#include <cerrno>
#include <climits>


/* Compare 100 staged and 100 device-buffer exchanges after two warm-ups each.
 * Both paths use Sendrecv_replace, matching examples 02 and 03. Device-buffer
 * MPI requires CUDA-aware support; it does not guarantee GPUDirect transport.
 */
__global__ void fill(float *buffer, int count, float value)
{
  size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < (size_t)count) {
    buffer[i] = value;
  }
}


void exchange(float *device, float *host, int count, int peer, bool staged)
{
  size_t bytes = (size_t)count * sizeof(float);
  if (staged) {
    CUDA_CHECK(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
  }
  MPI_CHECK(MPI_Sendrecv_replace(
      staged ? host : device, /* same buffer for outgoing and incoming values */
      count,                  /* number of floats */
      MPI_FLOAT,              /* element type */
      peer,                   /* destination rank */
      13,                     /* send tag */
      peer,                   /* source rank */
      13,                     /* receive tag */
      MPI_COMM_WORLD,         /* both participating ranks */
      MPI_STATUS_IGNORE       /* receive status is not needed */
  ));
  if (staged) {
    CUDA_CHECK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
  }
}


int main(int argc, char **argv)
{
  MPI_CHECK(MPI_Init(&argc, &argv));
  int rank, size;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));


  long parsed = 1 << 20;
  char *end = nullptr;
  errno = 0;
  if (argc == 2) {
    parsed = std::strtol(argv[1], &end, 10);
  }
  if (size != 2 || argc > 2 || errno == ERANGE || parsed < 1 ||
      parsed > INT_MAX || (argc == 2 && (end == argv[1] || *end != '\0'))) {
    if (!rank) {
      std::fprintf(stderr, "Usage: mpirun -np 2 07-warmup-run [float-count: 1..INT_MAX]\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
    return 1;
  }
  int count = (int)parsed;
  select_device(MPI_COMM_WORLD);


  size_t bytes = (size_t)count * sizeof(float);
  float *device, *host;
  CUDA_CHECK(cudaMalloc(&device, bytes));
  CUDA_CHECK(cudaMallocHost(&host, bytes));
  const int warmups = 2;
  const int runs = 100;
  int any_failure = 0;


  for (int mode = 0; mode < 2; ++mode) {
    bool staged = mode == 0;
    /* Reset both methods to the same data; finish initialization before timing. */
    fill<<<((size_t)count + 255) / 256, 256>>>(device, count, (float)rank);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    for (int i = 0; i < warmups; ++i) {
      exchange(device, host, count, 1 - rank, staged);
    }


    /* Time the full batch, then report the average time per exchange.
     * Blocking MPI and blocking staging copies establish completion here.
     * The barrier, reductions, and result validation are outside the timer.
     */
    MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
    double start = MPI_Wtime();
    for (int i = 0; i < runs; ++i) {
      exchange(device, host, count, 1 - rank, staged);
    }
    double local_time = MPI_Wtime() - start;
    double elapsed;
    MPI_CHECK(MPI_Reduce(&local_time, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                         MPI_COMM_WORLD));


    /* Check every received element outside the timed region on both ranks. */
    CUDA_CHECK(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
    /* Each exchange swaps the contents; an even total restores our rank. */
    float expected = (float)(((warmups + runs) % 2 == 0) ? rank : 1 - rank);
    int failed = 0;
    for (int i = 0; i < count; ++i) {
      if (host[i] != expected) {
        failed = 1;
        break;
      }
    }
    int global_failure;
    MPI_CHECK(MPI_Allreduce(&failed, &global_failure, 1, MPI_INT, MPI_MAX,
                            MPI_COMM_WORLD));
    any_failure |= global_failure;
    if (!rank) {
      std::printf("%s: %d floats, %.6f MiB/rank, 2 warm-ups, "
                  "%d timed exchanges, average %.3f ms/exchange, final sample %.0f, validation %s\n",
                  staged ? "staged" : "device-direct", count,
                  bytes / 1048576.0, runs, elapsed * 1e3 / runs, host[0],
                  global_failure ? "FAIL" : "PASS");
    }
  }


  CUDA_CHECK(cudaFreeHost(host));
  CUDA_CHECK(cudaFree(device));
  MPI_CHECK(MPI_Finalize());
  return any_failure ? 1 : 0;
}
