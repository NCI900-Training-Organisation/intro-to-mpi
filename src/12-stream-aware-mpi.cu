#include "common.h"

/*
 * Demonstrate explicit CUDA stream ordering around CUDA-aware MPI.
 *
 * MPI does not automatically observe arbitrary CUDA stream dependencies. The
 * producer kernel is recorded on a non-default stream, then an event is used
 * to establish that the device buffer is ready before MPI reads it. After the
 * blocking MPI_Sendrecv returns, the consumer kernel can use the received
 * buffer directly.
 *
 * This example uses MPI_Sendrecv so the focus stays on CUDA/MPI ordering. The
 * exact stream-aware behavior of a nonblocking MPI implementation varies; the
 * explicit host-side synchronization shown here is the portable baseline.
 */

__global__ void fill(double *buffer, int count, double value)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    buffer[index] = value;
  }
}


__global__ void increment(double *buffer, int count)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < count) {
    buffer[index] += 1.0;
  }
}


int main(int argc, char **argv)
{
  MPI_CHECK(MPI_Init(&argc, &argv));

  int rank, size;
  MPI_CHECK(MPI_Comm_rank(MPI_COMM_WORLD, &rank));
  MPI_CHECK(MPI_Comm_size(MPI_COMM_WORLD, &size));
  select_device(MPI_COMM_WORLD);

  int count = argc > 1 ? atoi(argv[1]) : 1 << 20;
  size_t bytes = (size_t)count * sizeof(double);
  double *send_buffer, *receive_buffer;
  CUDA_CHECK(cudaMalloc(&send_buffer, bytes));
  CUDA_CHECK(cudaMalloc(&receive_buffer, bytes));

  cudaStream_t producer_stream, consumer_stream;
  cudaEvent_t ready_for_mpi;

  /* Nonblocking streams avoid implicit synchronization with the legacy default
   * stream; CUDA/MPI dependencies must still be synchronized explicitly. */
  /* Run the kernel that fills the send buffer on the producer stream. */
  CUDA_CHECK(cudaStreamCreateWithFlags(&producer_stream, cudaStreamNonBlocking));
  /* Run the kernel that uses the received buffer on the consumer stream. */
  CUDA_CHECK(cudaStreamCreateWithFlags(&consumer_stream, cudaStreamNonBlocking));
  /* This event is used for ordering only, so disable timestamp collection.
   * Record ready_for_mpi after the producer kernel; the host waits on it before
   * passing the send buffer to MPI. Creating an event does not record it. */
  CUDA_CHECK(cudaEventCreateWithFlags(&ready_for_mpi, cudaEventDisableTiming));

  int peer = (size == 1) ? MPI_PROC_NULL : (rank + 1) % size;
  int source = (size == 1) ? MPI_PROC_NULL : (rank + size - 1) % size;
  double value = (double)rank;

  fill<<<(count + 255) / 256, 256, 0, producer_stream>>>(
      send_buffer, count, value);
  CUDA_CHECK(cudaEventRecord(ready_for_mpi, producer_stream));

  /* The host waits for the producer stream before MPI accesses the buffer. */
  CUDA_CHECK(cudaEventSynchronize(ready_for_mpi));
  MPI_CHECK(MPI_Sendrecv(
      send_buffer,                    /* device buffer produced by the kernel */
      count,                          /* number of elements sent */
      MPI_DOUBLE,                     /* element datatype */
      peer,                           /* destination rank */
      12,                             /* send tag */
      receive_buffer,                 /* device buffer filled by MPI */
      count,                          /* number of elements received */
      MPI_DOUBLE,                     /* receive datatype */
      source,                          /* source rank */
      12,                             /* receive tag */
      MPI_COMM_WORLD,                 /* communicator */
      MPI_STATUS_IGNORE               /* status is not needed here */
  ));

  /* Blocking MPI_Sendrecv has completed the receive, so no CUDA event is
   * needed before launching the consumer kernel. */
  increment<<<(count + 255) / 256, 256, 0, consumer_stream>>>(
      receive_buffer, count);
  CUDA_CHECK(cudaStreamSynchronize(consumer_stream));

  double sample = 0.0;
  CUDA_CHECK(cudaMemcpy(&sample, receive_buffer, sizeof(sample),
                        cudaMemcpyDeviceToHost));
  if (!rank) {
    printf("stream-aware MPI: %d ranks, %d values, first received value %.1f\n",
           size, count, sample);
  }

  CUDA_CHECK(cudaEventDestroy(ready_for_mpi));
  CUDA_CHECK(cudaStreamDestroy(producer_stream));
  CUDA_CHECK(cudaStreamDestroy(consumer_stream));
  CUDA_CHECK(cudaFree(send_buffer));
  CUDA_CHECK(cudaFree(receive_buffer));
  MPI_CHECK(MPI_Finalize());
  return 0;
}
