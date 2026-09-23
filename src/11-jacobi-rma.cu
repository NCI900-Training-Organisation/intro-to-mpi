#include "00-common.h"
#include <algorithm>

/*
 * Jacobi heat solver using MPI one-sided communication (RMA).
 *
 * Each rank owns a horizontal strip of the global domain and stores two ghost
 * rows around its real rows. Unlike the blocking and overlap examples, this
 * version does not use MPI_Sendrecv or matching receives. Before each stencil
 * update, MPI_Get reads the neighboring ranks' boundary rows directly into
 * this rank's device ghost rows.
 *
 * The MPI windows expose device allocations. This is an advanced CUDA-aware
 * MPI example: the MPI implementation must support the selected RMA operation
 * for device buffers, and it may use direct GPU transport or internal staging.
 * Passing a device pointer does not prove that a GPUDirect path was used.
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
  MPI_CHECK(MPI_Init(&argc, &argv));

  int rank, size;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
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

  int rows = ny / size;
  int first = rank * rows;
  int up = rank ? rank - 1 : MPI_PROC_NULL;
  int down = rank < size - 1 ? rank + 1 : MPI_PROC_NULL;

  size_t bytes = (size_t)(rows + 2) * nx * sizeof(double);
  double *u, *v;
  CUDA_CHECK(cudaMalloc(&u, bytes));
  CUDA_CHECK(cudaMalloc(&v, bytes));
  CUDA_CHECK(cudaMemset(v, 0, bytes));

  dim3 block(32, 8);
  dim3 grid((nx + 31) / 32, (rows + 9) / 8);
  init<<<grid, block>>>(u, rows, nx, ny, first);
  init<<<grid, block>>>(v, rows, nx, ny, first);
  CUDA_CHECK(cudaDeviceSynchronize());

  /*
   * An MPI window is the remotely accessible region of memory used by RMA
   * operations such as MPI_Get and MPI_Put. Each rank contributes its local
   * device allocation to the same window object, so a rank can read a row from
   * a neighbor without that neighbor posting a matching receive.
   *
   * MPI_Win_create takes the size in bytes and the displacement unit used by
   * later RMA calls. Using sizeof(double) means a target displacement of nx
   * refers to nx doubles, which makes row offsets easy to express. The window
   * itself does not copy the data; it describes the already allocated buffer.
   *
   * Jacobi alternates between u and v after every iteration. Because an MPI
   * window is attached to one allocation, both device buffers need their own
   * window. The current iteration selects the matching window below.
   */
  MPI_Win win_u, win_v;
  MPI_Aint window_bytes = static_cast<MPI_Aint>(bytes);
    MPI_CHECK(MPI_Win_create(
      u,                                  /* local device buffer to expose */
      window_bytes,                       /* exposed allocation size in bytes */
      sizeof(double),                     /* target displacements count doubles */
      MPI_INFO_NULL,                      /* no implementation-specific hints */
      MPI_COMM_WORLD,                     /* all ranks expose matching windows */
      &win_u                              /* handle for the u allocation */
    ));
    MPI_CHECK(MPI_Win_create(
      v,                                  /* second local device buffer */
      window_bytes,                       /* same allocation size as u */
      sizeof(double),                     /* target displacements count doubles */
      MPI_INFO_NULL,                      /* no implementation-specific hints */
      MPI_COMM_WORLD,                     /* all ranks expose matching windows */
      &win_v                              /* handle for the v allocation */
    ));
    MPI_CHECK(MPI_Win_lock_all(
      0,                                  /* assert no special lock mode */
      win_u                               /* start an access epoch on u */
    ));
    MPI_CHECK(MPI_Win_lock_all(
      0,                                  /* assert no special lock mode */
      win_v                               /* start an access epoch on v */
    ));

  MPI_CHECK(MPI_Barrier(MPI_COMM_WORLD));
  double start = MPI_Wtime();

  int begin = (rank == 0) ? 2 : 1;
  int end = (rank == size - 1) ? rows - 1 : rows;
  dim3 work((nx + 31) / 32, (end - begin + 8) / 8);

  for (int k = 0; k < iters; ++k) {
    double *current = (k % 2 == 0) ? u : v;
    double *next = (k % 2 == 0) ? v : u;
    MPI_Win current_window = (k % 2 == 0) ? win_u : win_v;

    /*
     * The origin is this rank's ghost row. The target displacement is a row
     * in the neighbor's exposed buffer, measured in MPI_DOUBLE elements:
     *   up rank's last real row:   rows * nx
     *   down rank's first real row: nx
     * MPI_Get is a remote read, so the neighbor does not call a matching MPI
     * receive. MPI_Win_flush_all completes the reads and makes the received
     * ghost data available at the origin before the CUDA stencil starts.
     */
    if (up != MPI_PROC_NULL) {
      /* Read the upper neighbor's last real row into this rank's top halo. */
      MPI_CHECK(MPI_Get(
          current,                 /* origin: this rank's top ghost row */
          nx,                      /* number of values in the origin row */
          MPI_DOUBLE,              /* datatype of the origin values */
          up,                      /* target rank above this rank */
          rows * nx,               /* target displacement: target's last row */
          nx,                      /* number of values to read remotely */
          MPI_DOUBLE,              /* datatype of the target values */
          current_window           /* window exposing the current buffer */
      ));
    }
    if (down != MPI_PROC_NULL) {
      /* Read the lower neighbor's first real row into this rank's bottom halo. */
      MPI_CHECK(MPI_Get(
          current + (rows + 1) * nx, /* origin: this rank's bottom ghost row */
          nx,                        /* number of values in the origin row */
          MPI_DOUBLE,                /* datatype of the origin values */
          down,                      /* target rank below this rank */
          nx,                        /* target displacement: target's first row */
          nx,                        /* number of values to read remotely */
          MPI_DOUBLE,                /* datatype of the target values */
          current_window             /* window exposing the current buffer */
      ));
    }
    /*
     * MPI_Get starts the remote reads but may return before the ghost rows are
     * ready. Flush the active window to wait for all outstanding RMA operations
     * to complete and make their data available in this rank's device buffer.
     * The stencil is launched only after this completion point.
     */
    MPI_CHECK(MPI_Win_flush_all(
      current_window /* window whose outstanding MPI_Get operations finish */
    ));
    CUDA_CHECK(cudaDeviceSynchronize());

    step<<<work, block>>>(current, next, nx, begin, end);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  double local = MPI_Wtime() - start;
  double elapsed;
  MPI_CHECK(MPI_Reduce(&local, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0,
                       MPI_COMM_WORLD));

  double *result = (iters % 2 == 0) ? u : v;
  double sample;
  CUDA_CHECK(cudaMemcpy(&sample, result + (rows / 2) * nx + nx / 2,
                        sizeof(double), cudaMemcpyDeviceToHost));
  double checksum;
  MPI_CHECK(MPI_Reduce(&sample, &checksum, 1, MPI_DOUBLE, MPI_SUM, 0,
                       MPI_COMM_WORLD));

  if (!rank) {
    printf("RMA Jacobi: %dx%d, %d ranks, %d iterations, %.3f s, sample "
           "sum %.12e\n",
           nx, ny, size, iters, elapsed, checksum);
  }

  MPI_CHECK(MPI_Win_unlock_all(win_u));
  MPI_CHECK(MPI_Win_unlock_all(win_v));
  MPI_CHECK(MPI_Win_free(&win_u));
  MPI_CHECK(MPI_Win_free(&win_v));
  CUDA_CHECK(cudaFree(u));
  CUDA_CHECK(cudaFree(v));

  MPI_CHECK(MPI_Finalize());
  return 0;
}
