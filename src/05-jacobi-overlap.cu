#include "00-common.h"
#include <algorithm>

/*
 * Overlapped Jacobi stencil.
 *
 * This version is the same as the blocking example in 04-jacobi-blocking.cu in
 * its domain layout, ghost rows, and stencil update; the main difference is the
 * communication pattern. Instead of blocking on MPI_Sendrecv(), we post
 * nonblocking receives and sends for the halo rows, start the interior update
 * immediately, and then wait for the communication to finish before updating the
 * first and last local rows. This overlaps communication with computation.
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


__global__ void step_rows(const double *u, double *v, int nx, int begin,
                          int end)
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


  int nx = argc > 1 ? atoi(argv[1]) : 4096,
      ny = argc > 2 ? atoi(argv[2]) : 4096,
      iters = argc > 3 ? atoi(argv[3]) : 200;


  if (ny % size || nx < 3 || ny / size < 3) {
    if (!rank) {
      fprintf(stderr, "ny must divide by ranks and provide >=3 rows/rank\n");
    }
    MPI_Abort(MPI_COMM_WORLD, 1);
  }


  int rows = ny / size, first = rank * rows;
  int up = rank ? rank - 1 : MPI_PROC_NULL;
  int down = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;


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


  for (int k = 0; k < iters; k++) {
    /*
     * Nonblocking halo exchange. Unlike the blocking version, these requests are
     * posted first and the interior work is launched before the communication has
     * necessarily completed. This allows overlap between MPI traffic and the GPU
     * kernel that updates the interior rows.
     */
    MPI_Request q[4];
    MPI_CHECK(MPI_Irecv(
        u,                                    /* top ghost row receive buffer from rank above */
        nx,                                   /* one row of nx doubles */
        MPI_DOUBLE,                           /* data type */
        up,                                   /* source rank above */
        11,                                   /* tag for top halo receive */
        MPI_COMM_WORLD,                       /* communicator */
        &q[0]                                 /* request handle for this receive */
    ));
    MPI_CHECK(MPI_Irecv(
        u + (rows + 1) * nx,                  /* bottom ghost row receive buffer from rank below */
        nx,                                   /* one row of nx doubles */
        MPI_DOUBLE,                           /* data type */
        down,                                 /* source rank below */
        10,                                   /* tag for bottom halo receive */
        MPI_COMM_WORLD,                       /* communicator */
        &q[1]                                 /* request handle for this receive */
    ));
    MPI_CHECK(MPI_Isend(
        u + nx,                               /* first real row to send upward */
        nx,                                   /* one row of nx doubles */
        MPI_DOUBLE,                           /* data type */
        up,                                   /* destination rank above */
        10,                                   /* tag for upward send */
        MPI_COMM_WORLD,                       /* communicator */
        &q[2]                                 /* request handle for this send */
    ));
    MPI_CHECK(MPI_Isend(
        u + rows * nx,                        /* last real row to send downward */
        nx,                                   /* one row of nx doubles */
        MPI_DOUBLE,                           /* data type */
        down,                                 /* destination rank below */
        11,                                   /* tag for downward send */
        MPI_COMM_WORLD,                       /* communicator */
        &q[3]                                 /* request handle for this send */
    ));


    /*
     * Interior rows can be updated while the halo exchange is still in flight.
     * The kernel starts from row 2 and ends at row rows - 1, which excludes the
     * top/bottom ghost rows and the first/last owned boundary rows handled next.
     */
    dim3 gi((nx + 31) / 32, (rows - 2 + 7) / 8);
    step_rows<<<gi, b>>>(u, v, nx, 2, rows - 1);


    /*
     * Wait for the halo exchange to complete before updating the two rows that
     * directly touch the ghost data. This keeps the stencil correct while still
     * overlapping most of the communication with the interior compute.
     */
    MPI_CHECK(MPI_Waitall(4, q, MPI_STATUSES_IGNORE));


    /*
     * Update the first and last owned row separately once the neighboring halo
     * data is available. The global-domain edges are skipped with rank checks.
     */
    dim3 ge((nx + 31) / 32, 1);
    if (rank > 0) {
      step_rows<<<ge, b>>>(u, v, nx, 1, 1);
    }
    if (rank < size - 1) {
      step_rows<<<ge, b>>>(u, v, nx, rows, rows);
    }


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
    printf("overlap Jacobi: %dx%d, %d ranks, %d iterations, %.3f s, sample sum "
           "%.12e\n",
           nx, ny, size, iters, elapsed, checksum);
  }


  cudaFree(u);
  cudaFree(v);


  MPI_Finalize();
  return 0;
}
