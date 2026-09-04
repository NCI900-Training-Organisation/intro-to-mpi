Requested-reference coverage map
================================

The requested NVIDIA, Intel, and EPCC materials were used as a minimum topic
check. This repository's examples are independently written for CUDA/Open MPI
on Gadi; Intel concepts are mapped without pretending Gadi has Intel GPUs.

.. list-table::
   :header-rows: 1
   :widths: 34 40 26

   * - Topic
     - Course chapter
     - Executable example
   * - MPI ranks and local-rank GPU assignment
     - :doc:`08-mpi-fundamentals`, :doc:`01-setup`
     - ``01``, ``10``
   * - Explicit pinned-host staging
     - :doc:`02-staging`, :doc:`09-cuda-memory`
     - ``02``
   * - Device pointers and UVA
     - :doc:`03-cuda-aware`, :doc:`09-cuda-memory`
     - ``03``
   * - GPUDirect P2P, RDMA, fallback pipelining, and topology
     - :doc:`11-gpudirect-topology`
     - ``03``, ``06`` (adapt placement for scale-out)
   * - Blocking and nonblocking point-to-point MPI
     - :doc:`12-communication-patterns`
     - ``03``–``05``
   * - CUDA/MPI ordering, ownership, streams, and progress
     - :doc:`10-synchronization`
     - ``03``–``10``
   * - Jacobi halo exchange and overlap
     - :doc:`04-jacobi`, :doc:`05-overlap`
     - ``04``, ``05``
   * - Message-size benchmarking
     - :doc:`15-performance-methodology`
     - ``06``
   * - GPU-buffer collectives
     - :doc:`13-collectives`
     - ``07``
   * - Noncontiguous data and MPI datatypes
     - :doc:`09-cuda-memory`
     - ``08``
   * - Managed/unified memory
     - :doc:`09-cuda-memory`
     - ``09``
   * - Device selection before ``MPI_Init``
     - :doc:`14-debugging-portability`
     - ``10``
   * - Intel MPI execution models, topology, and pinning
     - :doc:`14-debugging-portability`
     - portability discussion
   * - Profiling, correctness, and reproducibility
     - :doc:`14-debugging-portability`, :doc:`15-performance-methodology`
     - ``06`` and paired ``04``/``05`` runs

Primary references
------------------

* `NVIDIA: An Introduction to CUDA-Aware MPI <https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/>`_
* `Intel: Running MPI Applications on GPUs <https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-0/running-mpi-applications-on-gpus.html>`_
* `EPCC: Using MPI with GPUs <https://epcced.github.io/archer-gpu-course/section-5.02/>`_
