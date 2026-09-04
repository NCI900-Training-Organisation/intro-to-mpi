Synchronization, streams, and progress
========================================

CUDA kernels and memory copies are commonly asynchronous with respect to the
CPU. MPI completion and CUDA completion are separate events. Correct programs
make the dependency between them explicit.

A safe blocking sequence
------------------------

For a buffer produced by a kernel:

.. code-block:: text

   launch kernel -> CUDA completion -> MPI send
   MPI receive -> MPI completion -> CUDA kernel consumes buffer

``cudaDeviceSynchronize`` is easy to teach and debug, but it waits for all work
on the current device. A CUDA event or stream-specific synchronization can
provide narrower dependencies when the MPI implementation and application
make that safe.

Streams and nonblocking MPI
---------------------------

MPI generally cannot infer arbitrary CUDA stream dependencies. A
``cudaMemcpyAsync`` must be ordered with an event or stream synchronization
before MPI accesses its host buffer. Similarly, an MPI receive must complete
before a dependent kernel is launched. Do not assume that placing calls next
to each other creates a cross-runtime dependency.

Nonblocking operations
----------------------

``MPI_Isend`` and ``MPI_Irecv`` return requests that represent incomplete
operations. The application must call ``MPI_Wait``, ``MPI_Waitall``, ``MPI_Test``,
or a related operation before reusing the buffers. A request may complete from
the CPU's point of view while a CUDA-aware implementation still requires the
MPI-defined buffer ownership rules to be respected; follow the loaded MPI
implementation's documentation.

Progress
--------

Nonblocking does not automatically mean that communication progresses while a
GPU kernel runs. Progress may depend on message size, transport, MPI progress
threads, CPU availability, and calls back into MPI. Measure the actual stack.
A short interior kernel may finish before communication makes useful progress,
leaving no overlap to exploit.

Ordering checklist
------------------

* Establish producer-kernel completion before MPI reads a send buffer.
* Post receives before sends when that avoids a protocol deadlock.
* Do not overwrite a request's send buffer before completion.
* Wait for receives before launching kernels that consume them.
* Synchronize the stream that owns the relevant CUDA work, not an unrelated
  stream.
* Treat ``MPI_Barrier`` as coordination, not a substitute for buffer
  ownership or CUDA synchronization.
