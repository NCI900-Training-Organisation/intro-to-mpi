#include "00-common.h"
#include <string>

__global__ void initialise(double *a, int rows, int cols, int rank)
{
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < rows * cols)
    a[i] = rank * 1000000.0 + i;
}
__global__ void pack_column(const double *a, double *packed, int rows, int cols,
                            int column)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows)
    packed[row] = a[row * cols + column];
}
__global__ void unpack_column(double *a, const double *packed, int rows,
                              int cols, int column)
{
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows)
    a[row * cols + column] = packed[row];
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
  const int rows = 1024, cols = 1024, peer = 1 - rank;
  bool datatype = argc > 1 && std::string(argv[1]) == "datatype";
  double *matrix;
  CUDA_CHECK(cudaMalloc(&matrix, (size_t)rows * cols * sizeof(double)));
  initialise<<<(rows * cols + 255) / 256, 256>>>(matrix, rows, cols, rank);
  CUDA_CHECK(cudaDeviceSynchronize());
  if (datatype) {
    MPI_Datatype column;
    MPI_CHECK(MPI_Type_vector(rows, 1, cols, MPI_DOUBLE, &column));
    MPI_CHECK(MPI_Type_commit(&column));
    MPI_CHECK(MPI_Sendrecv(matrix + cols - 2, 1, column, peer, 8,
                           matrix + cols - 1, 1, column, peer, 8,
                           MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    MPI_CHECK(MPI_Type_free(&column));
  } else {
    double *send, *recv;
    CUDA_CHECK(cudaMalloc(&send, rows * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&recv, rows * sizeof(double)));
    pack_column<<<(rows + 255) / 256, 256>>>(matrix, send, rows, cols,
                                             cols - 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_CHECK(MPI_Sendrecv(send, rows, MPI_DOUBLE, peer, 8, recv, rows,
                           MPI_DOUBLE, peer, 8, MPI_COMM_WORLD,
                           MPI_STATUS_IGNORE));
    unpack_column<<<(rows + 255) / 256, 256>>>(matrix, recv, rows, cols,
                                               cols - 1);
    CUDA_CHECK(cudaFree(send));
    CUDA_CHECK(cudaFree(recv));
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  double value;
  CUDA_CHECK(cudaMemcpy(&value, matrix + cols - 1, sizeof(value),
                        cudaMemcpyDeviceToHost));
  double expected = peer * 1000000.0 + cols - 2;
  bool ok = value == expected;
  std::printf("rank %d: %s strided halo %.1f (expected %.1f) %s\n", rank,
              datatype ? "datatype" : "packed", value, expected,
              ok ? "PASS" : "FAIL");
  CUDA_CHECK(cudaFree(matrix));
  MPI_Finalize();
  return ok ? 0 : 1;
}
