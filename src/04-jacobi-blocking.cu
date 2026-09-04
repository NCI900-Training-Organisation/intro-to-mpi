#include "00-common.h"
#include <algorithm>

__global__ void init(double *u, int rows, int nx, int ny, int first) {
  int x = blockIdx.x * blockDim.x + threadIdx.x,
      y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx && y < rows + 2) {
    int gy = first + y - 1;
    u[y * nx + x] =
        (x == 0 || x == nx - 1 || gy == 0 || gy == ny - 1) ? 1. : 0.;
  }
}

__global__ void step(const double *u, double *v, int nx, int begin, int end) {
  int x = blockIdx.x * blockDim.x + threadIdx.x + 1,
      y = blockIdx.y * blockDim.y + threadIdx.y + begin;
  if (x < nx - 1 && y <= end) {
    v[y * nx + x] = .25 * (u[y * nx + x - 1] + u[y * nx + x + 1] +
                           u[(y - 1) * nx + x] + u[(y + 1) * nx + x]);
  }
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int rank, size;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  select_device(MPI_COMM_WORLD);

  int nx = argc > 1 ? atoi(argv[1]) : 4096,
      ny = argc > 2 ? atoi(argv[2]) : 4096,
      iters = argc > 3 ? atoi(argv[3]) : 200;
  if (ny % size || nx < 3 || ny / size < 3) {
    if (!rank) {
      fprintf(stderr, "ny must divide by ranks and provide >=3 rows/rank\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  int rows = ny / size, first = rank * rows,
      up = rank ? rank - 1 : MPI_PROC_NULL,
      down = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;
  size_t bytes = (size_t)(rows + 2) * nx * sizeof(double);
  double *u, *v;

  CUDA_CHECK(cudaMalloc(&u, bytes));
  CUDA_CHECK(cudaMalloc(&v, bytes));
  CUDA_CHECK(cudaMemset(v, 0, bytes));
  dim3 b(32, 8), g((nx + 31) / 32, (rows + 9) / 8);
  init<<<g, b>>>(u, rows, nx, ny, first);
  init<<<g, b>>>(v, rows, nx, ny, first);
  CUDA_CHECK(cudaDeviceSynchronize());

  MPI_Barrier(MPI_COMM_WORLD);
  double t = MPI_Wtime();
  int begin = rank == 0 ? 2 : 1, end = rank == size - 1 ? rows - 1 : rows;
  dim3 work((nx + 31) / 32, (end - begin + 8) / 8);
  for (int k = 0; k < iters; k++) {
    MPI_CHECK(MPI_Sendrecv(u + nx, nx, MPI_DOUBLE, up, 10, u + (rows + 1) * nx,
                           nx, MPI_DOUBLE, down, 10, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE));
    MPI_CHECK(MPI_Sendrecv(u + rows * nx, nx, MPI_DOUBLE, down, 11, u, nx,
                           MPI_DOUBLE, up, 11, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE));
    step<<<work, b>>>(u, v, nx, begin, end);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::swap(u, v);
  }

  double local = MPI_Wtime() - t, elapsed;
  MPI_Reduce(&local, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  double sample;
  CUDA_CHECK(cudaMemcpy(&sample, u + (rows / 2) * nx + nx / 2, sizeof(double),
                        cudaMemcpyDeviceToHost));
  double checksum;
  MPI_Reduce(&sample, &checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if (!rank) {
    printf("blocking Jacobi: %dx%d, %d ranks, %d iterations, %.3f s, sample "
           "sum %.12e\n",
           nx, ny, size, iters, elapsed, checksum);
  }

  cudaFree(u);
  cudaFree(v);
  MPI_Finalize();
  return 0;
}
