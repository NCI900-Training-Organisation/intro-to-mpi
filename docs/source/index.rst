CUDA-aware MPI 
======================

This documentation is an introductory guide to using MPI with CUDA GPUs.

.. note::

   The examples target NCI Gadi, project ``vp91``, queue ``gpuvolta``, and
   NVIDIA Volta GPUs. 
  

  Participants should know basic C/C++, CUDA kernels, and MPI point-to-point
  calls.


Learning outcomes
-----------------

By the end you can 

 #. Map one MPI rank to each GPU

 #. Pass device pointers to MPI

 #. Describe UVA and GPUDirect paths

 #. Manage CUDA/MPI synchronisation


 #. Overlap GPU computation with communication

Topics Covered
-------------------

.. list-table::
   :header-rows: 1

   * - Section
     - Hands-on result
   * - :doc:`01-setup`
     - Correct rank-to-GPU placement
   * - :doc:`02-staging`
     - Explicit host-staged transfer
   * - :doc:`03-cuda-aware`
     - Direct device-buffer transfer
   * - :doc:`04-jacobi`
     - Distributed blocking solver
   * - :doc:`05-overlap`
     - Nonblocking overlapped solver
   * - :doc:`20-cuda-stream-aware-mpi`
     - Explicit CUDA stream ordering around MPI
   * - :doc:`18-cuda-aware-mpi-diagnostics`
     - Investigate CUDA-aware support and transport evidence
   * - :doc:`21-gpu-aware-mpi-best-practices`
     - Apply correctness and performance checks

.. toctree::
   :maxdepth: 1
   :caption: Core workshop
   :numbered:

   01-setup
   02-staging
   03-cuda-aware
   04-jacobi
   05-overlap
   20-cuda-stream-aware-mpi
   21-gpu-aware-mpi-best-practices
   07-reference

..
   .. toctree::
      :maxdepth: 2
      :caption: Extended topics and reference
      :numbered:

      06-performance
      07-reference
      08-mpi-fundamentals
      09-cuda-memory
      10-synchronization
      11-gpudirect-topology
      12-communication-patterns
      13-collectives
      14-debugging-portability
      15-performance-methodology
      16-alternatives-and-future
      17-coverage-map
      18-cuda-aware-mpi-diagnostics
      19-mpi-rma-gpu
      
