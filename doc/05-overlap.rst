Nonblocking exchange and overlap
================================

The optimised loop posts two receives and two sends, computes rows that do not
depend on incoming halos, waits for communication, then computes the two edge
rows. Submit it with:

.. code-block:: console

   $ qsub jobs-scripts/05-jacobi-overlap.pbs

.. code-block:: text

   post Irecv/Isend -> interior CUDA kernel -> MPI_Waitall -> edge kernels
          communication  <---- potential overlap ---->  computation

Nonblocking is necessary but not sufficient. Progress can depend on the MPI
implementation, message size, and whether calls into MPI are needed while the
kernel runs. The GPU kernel launch is asynchronous; ``MPI_Waitall`` gives MPI a
chance to progress while the interior kernel executes. A final CUDA
synchronisation protects the array swap.

Avoid these common errors
-------------------------

Do not reuse send rows before ``MPI_Waitall``. Post receives before sends. Use
unique, consistently matched tags. Do not run edge kernels until receives have
completed. Keep messages contiguous unless the installed stack's datatype path
has been validated. CUDA events time GPU work; ``MPI_Wtime`` plus rank barriers
is appropriate for end-to-end distributed time.

.. admonition:: Exercise (25 minutes)
   :class: exercise

   Predict when overlap helps by comparing halo bytes (``2*nx*sizeof(double)``)
   with interior work. Run blocking and overlap versions for 2048 and 8192.
   Explain a slowdown if the interior kernel is too short to hide communication.
