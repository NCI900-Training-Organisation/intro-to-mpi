CUDA stream-aware MPI integration
=================================

CUDA-aware MPI accepts device pointers, but MPI does not automatically know
that a CUDA kernel running in a non-default stream must finish before MPI reads
the buffer. The application must make the producer-to-MPI dependency explicit.

The companion example ``src/12-stream-aware-mpi.cu`` uses a CUDA event and
streams around a simple ``MPI_Sendrecv``. It demonstrates the conservative
portable rule: record an event after the producer kernel, wait for that event
before MPI accesses the device buffer, launch the consumer kernel after
blocking MPI completes, and synchronize the consumer stream before reading
the result on the host.

Producer ordering
-----------------

The producer kernel runs on a non-default stream:

.. code-block:: c++

   fill<<<grid, block, 0, producer_stream>>>(send_buffer, count, value);
   cudaEventRecord(ready_for_mpi, producer_stream);
   cudaEventSynchronize(ready_for_mpi);
   MPI_Send(send_buffer, count, MPI_DOUBLE, peer, tag, MPI_COMM_WORLD);

The event is a host-visible proof that the kernel's writes are complete. Without
it, MPI may inspect a buffer while the GPU is still writing it.

Receive ordering
----------------

With a CUDA-aware MPI implementation that supports these device-buffer
operations, a blocking ``MPI_Sendrecv`` completes the receive before returning
to the calling CPU thread. The consumer kernel can therefore be launched
directly after the call:

.. code-block:: c++

   MPI_Sendrecv(send_buffer, count, MPI_DOUBLE, peer, tag,
                receive_buffer, count, MPI_DOUBLE, source, tag,
                MPI_COMM_WORLD, MPI_STATUS_IGNORE);
   increment<<<grid, block, 0, consumer_stream>>>(receive_buffer, count);
   cudaStreamSynchronize(consumer_stream);

No additional CUDA event is needed between MPI completion and the kernel
launch. Recording an event after MPI returns does not track MPI's internal
work; the MPI call itself establishes receive completion. The final
``cudaStreamSynchronize`` waits for the consumer kernel before the host reads
its result.

For a nonblocking receive, complete the request before launching the consumer:

.. code-block:: c++

   MPI_Irecv(receive_buffer, count, MPI_DOUBLE, source, tag,
             MPI_COMM_WORLD, &request);
   MPI_Wait(&request, MPI_STATUS_IGNORE);
   increment<<<grid, block, 0, consumer_stream>>>(receive_buffer, count);

The producer event remains necessary in this example because a CUDA kernel
launch returns before its GPU work finishes. In contrast, the blocking MPI
call or ``MPI_Wait`` returns only after the corresponding communication has
completed. Blocking the calling CPU thread does not stop unrelated GPU work.

What this does not guarantee
----------------------------

CUDA events establish ordering. They do not guarantee GPUDirect RDMA,
zero-copy transport, or asynchronous MPI progress. The MPI implementation may
still stage through pinned host memory, and a blocking MPI call may wait for
both transport completion and internal CUDA work.

Overlap requirements
--------------------

To overlap communication with GPU computation, verify all three dependencies:

* the producer stream has finished writing the send buffer;
* MPI owns the send buffer until the request completes; and
* the consumer stream waits until the receive buffer is valid.

Measure overlap with CUDA events for device work and ``MPI_Wtime`` for the
application timeline. A kernel running concurrently with an MPI call does not
prove useful overlap if the MPI library requires CPU progress or internally
serializes access to the device.

Portability checklist
---------------------

Before relying on stream-aware behavior, check the MPI implementation's CUDA
documentation and test the exact operation, buffer type, stream pattern, and
transport. The explicit producer event and blocking MPI completion in the companion
source is the baseline to compare against any implementation-specific
extension.