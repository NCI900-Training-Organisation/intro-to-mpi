#include "00-common.h"
#include <algorithm>

/*
 * Blocking Jacobi stencil on GPU-resident data.
 *
 * Each MPI rank owns a horizontal strip of the global 2-D domain. We pass a
 * device buffer with (rows + 2) * nx doubles so that each rank stores both the
 * real rows and a top/bottom halo row. The halo rows are filled by MPI
 * Sendrecv() before each Jacobi update, allowing the stencil to read the
 * neighboring rank's boundary values without doing any host staging.
 *
 * The stencil update is:
 *   v[i,j] = 0.25 * (u[i-1,j] + u[i+1,j] + u[i,j-1] + u[i,j+1])
 * and it is applied in-place on the device by alternating between u and v.
 */

__global__ void init(double *u, int rows, int nx, int ny, int first)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x,
      y = blockIdx.y * blockDim.y + threadIdx.y;
  if (x < nx && y < rows + 2) {
    int gy = first + y - 1;
    u[y * nx + x] =
        (x == 0 || x == nx - 1 || gy == 0 || gy == ny - 1) ? 1. : 0.;
  }
}


__global__ void step(const double *u, double *v, int nx, int begin, int end)
{
  int x = blockIdx.x * blockDim.x + threadIdx.x + 1,
      y = blockIdx.y * blockDim.y + threadIdx.y + begin;
  if (x < nx - 1 && y <= end) {
    v[y * nx + x] = .25 * (u[y * nx + x - 1] + u[y * nx + x + 1] +
                           u[(y - 1) * nx + x] + u[(y + 1) * nx + x]);
  }
}


int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);


  select_device(MPI_COMM_WORLD);


  int nx = argc > 1 ? atoi(argv[1]) : 4096;
  int ny = argc > 2 ? atoi(argv[2]) : 4096;
  int iters = argc > 3 ? atoi(argv[3]) : 200;


  if (ny % size || nx < 3 || ny / size < 3) {
    if (!rank) {
      fprintf(stderr, "ny must divide by ranks and provide >=3 rows/rank\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }


  /*
   * Domain decomposition: each rank owns a local chunk of height rows. The
   * global y index of local row y is first + y - 1, so the boundary condition
   * can be enforced correctly at the global domain edges.
   *
   * The ghost rows are stored as row 0 (top halo) and row rows + 1 (bottom
   * halo). The ranks above and below are connected through MPI ranks up/down.
   */
  int rows = ny / size;
  int first = rank * rows;
  int up = rank ? rank - 1 : MPI_PROC_NULL;
  int down = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;


  size_t bytes = (size_t)(rows + 2) * nx * sizeof(double);
  /*
   * u and v each hold one local domain slice, including the two ghost rows
   * needed for MPI halo exchange. They store alternating Jacobi states:
   * u = current iteration, v = next iteration.
   */
  double *u; 
  double *v; 
  CUDA_CHECK(cudaMalloc(&u, bytes));
  CUDA_CHECK(cudaMalloc(&v, bytes));


  CUDA_CHECK(cudaMemset(v, 0, bytes));
  /*
   * Thread block: 32 x 8 threads, so each block covers a tile of the local
   * domain. The launch grid rounds up the x/y dimensions to cover every point
   * in the (nx x (rows + 2)) device buffer, including the ghost rows.
   */
  dim3 b(32, 8); // Block dimension
  dim3 g((nx + 31) / 32, (rows + 9) / 8); // Grid dimension

  init<<<g, b>>>(u, rows, nx, ny, first);
  init<<<g, b>>>(v, rows, nx, ny, first);
  CUDA_CHECK(cudaDeviceSynchronize());


  MPI_Barrier(MPI_COMM_WORLD);
  double t = MPI_Wtime();


  int begin = (rank == 0) ? 2 : 1;
  int end = (rank == size - 1) ? rows - 1 : rows;
  dim3 work((nx + 31) / 32, (end - begin + 8) / 8);


  for (int k = 0; k < iters; k++) {
    /*
     * Halo exchange before every iteration. Each rank sends its top real row to
     * the rank above and receives the rank below's top row into the bottom
     * halo. The reverse exchange fills the top halo from the rank above.
     *
     * This blocking pattern is simple and correct, but it serializes the
     * communication and computation: the rank cannot start the next Jacobi step
     * until the ghost rows are fully exchanged.
     */
    MPI_CHECK(MPI_Sendrecv(
        u + nx,                               /* first real row to send upward */
        nx,                                   /* one full row of nx doubles */
        MPI_DOUBLE,                           /* data type */
        up,                                   /* rank above */
        10,                                   /* tag for upward exchange */
        u + (rows + 1) * nx,                  /* bottom ghost row to receive from below */
        nx,                                   /* one full row of nx doubles */
        MPI_DOUBLE,                           /* data type */
        down,                                 /* rank below */
        10,                                   /* matching tag for the below neighbor */
        MPI_COMM_WORLD,                       /* all ranks in the solver */
        MPI_STATUS_IGNORE                     /* ignore the MPI status */
    ));
    MPI_CHECK(MPI_Sendrecv(
        u + rows * nx,                        /* last real row to send downward */
        nx,                                   /* one full row of nx doubles */
        MPI_DOUBLE,                           /* data type */
        down,                                 /* rank below */
        11,                                   /* tag for downward exchange */
        u,                                    /* top ghost row to receive from above */
        nx,                                   /* one full row of nx doubles */
        MPI_DOUBLE,                           /* data type */
        up,                                   /* rank above */
        11,                                   /* matching tag for the above neighbor */
        MPI_COMM_WORLD,                       /* all ranks in the solver */
        MPI_STATUS_IGNORE                     /* ignore the MPI status */
    ));


    step<<<work, b>>>(u, v, nx, begin, end);
    CUDA_CHECK(cudaDeviceSynchronize());
    std::swap(u, v);
  }


  double local = MPI_Wtime() - t, elapsed;
  /*
   * local = time spent by this rank in the solve loop
   * elapsed = maximum time across all ranks, reported by rank 0
   */
  MPI_Reduce(&local, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);


  double sample;
  /*
   * Copy one representative value from the middle of the local domain from GPU
   * to host memory so it can be reduced across ranks.
   */
  CUDA_CHECK(cudaMemcpy(&sample, u + (rows / 2) * nx + nx / 2, sizeof(double),
                        cudaMemcpyDeviceToHost));
  double checksum;
  /*
   * checksum = sum of the sampled center value across all ranks.
   * This is a lightweight diagnostic, not a convergence check.
   */
  MPI_Reduce(&sample, &checksum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);


  if (!rank) {
    /*
     * Rank 0 prints the wall-clock time and the reduced sample checksum.
     */
    printf("blocking Jacobi: %dx%d, %d ranks, %d iterations, %.3f s, sample "
           "sum %.12e\n",
           nx, ny, size, iters, elapsed, checksum);
  }


  cudaFree(u);
  cudaFree(v);


  MPI_Finalize();
  return 0;
}
