Nonblocking exchange and overlap
================================

The optimised loop posts two receives and two sends, computes rows that do not
depend on incoming halos, waits for communication, then computes the two edge
rows. Submit it with:

.. code-block:: console

   $ qsub job-script/05-jacobi-overlap.pbs

.. code-block:: text

   post Irecv/Isend -> interior CUDA kernel -> MPI_Waitall -> edge kernels
      communication <------- potential overlap ---->  computation

Nonblocking is necessary but not sufficient. Progress can depend on the MPI
implementation, message size, and whether calls into MPI are needed while the
kernel runs. The GPU kernel launch is asynchronous; ``MPI_Waitall`` gives MPI a
chance to progress while the interior kernel executes. A final CUDA
synchronisation protects the array swap.

Avoid these common errors
-------------------------

* Do not reuse send rows before ``MPI_Waitall``. 

* Post receives before sends. 

* Use unique, consistently matched tags. 

* Do not run edge kernels until receives have completed. 



.. admonition:: Exercise (25 minutes)
   :class: exercise

   Review the code ``05-jacobi-overlap.cu`` and test it with different combination 
   of arguments. 

   How is the performance compared to blocked Jacobi application?
