Collectives and reductions with GPU buffers
=============================================

Collectives coordinate all ranks in a communicator. Common operations include
``MPI_Bcast``, ``MPI_Reduce``, ``MPI_Allreduce``, ``MPI_Gather``,
``MPI_Scatter``, and ``MPI_Alltoall``. CUDA-aware support is not guaranteed by
point-to-point support, so test each operation separately.

Device-buffer collectives
--------------------------

A device pointer may be accepted by one operation and rejected by another, or
may use an internal staging path. Check the loaded implementation and run a
small correctness test before placing a device-buffer collective in a solver's
critical path.

``07-device-collectives.cu`` is that test for ``MPI_Allreduce``. It creates its
input on the GPU, reduces into a second device allocation, copies back only for
verification, and prints ``PASS`` on every rank. Run it with
``qsub jobs-scripts/07-device-collectives.pbs`` before relying on collectives.

A portable reduction pattern is to compute one scalar per rank on the GPU,
copy that scalar to host memory, and reduce the host scalar:

.. code-block:: text

   GPU reduction -> device-to-host copy of one value -> MPI_Allreduce

The copy of one value is often negligible compared with reducing a large
array, and it avoids relying on device-buffer collective support. For larger
collectives, compare direct device buffers, explicit packing, and host staging
with realistic message sizes.

Numerical reductions
--------------------

Floating-point reductions are not generally bitwise reproducible because the
order of additions can change with rank count and tree shape. Use tolerances,
compensated or reproducible methods where required, and report the validation
criterion. For a Jacobi solver, useful checks include a global residual, a
norm, boundary invariants, and comparison with a small CPU reference.

Synchronization semantics
-------------------------

A collective is a communication operation, not a universal CUDA device
synchronization. Establish CUDA ordering before a device buffer is read, and
synchronize dependent device work after the collective according to the MPI
implementation's contract. A barrier does not make an unfinished kernel safe
to communicate.
