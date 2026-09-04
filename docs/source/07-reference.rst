Reference and troubleshooting
=============================

Command map
-----------

.. list-table::
   :header-rows: 1

   * - File
     - Concept
     - Ranks
   * - ``01-rank-device``
     - node-local rank and GPU affinity
     - 2 on one node
   * - ``02-staged-pingpong``
     - explicit pinned-host staging
     - exactly 2
   * - ``03-device-pingpong``
     - direct device pointer
     - exactly 2
   * - ``04-jacobi-blocking``
     - direct blocking halo exchange
     - 2 by supplied script
   * - ``05-jacobi-overlap``
     - nonblocking exchange and overlap
     - 2 by supplied script

Troubleshooting
---------------

**No CUDA device**
   Confirm the job requested ``ngpus`` and ran in ``gpuvolta``. Never test a
   CUDA executable on the login node.

**Compiler cannot find mpi.h**
   Load the pinned Open MPI module and confirm ``mpicxx --showme`` works.
   CMake propagates the MPI include paths and libraries through ``MPI::MPI_CXX``.

**Unknown module version**
   Run ``module avail cuda`` and ``module avail openmpi``. Update all numbered
   scripts together to a compatible, supported pair.

**Device-pointer MPI fails but staging works**
   Check ``ompi_info`` for CUDA support and consult NCI. Do not assume that
   every MPI module, transport, or MPI operation is CUDA-aware.

**Wrong answers or intermittent data**
   Check kernel completion before MPI, request completion before buffer reuse,
   correct local-rank affinity, message counts, tags, and ghost-row offsets.
   Run ``CUDA_LAUNCH_BLOCKING=1`` as a diagnostic, not a performance setting.

**PBS job cannot see the working directory**
   Ensure the ``storage`` directive names each required ``scratch`` or ``gdata``
   filesystem. Submit from the repository root because scripts use relative paths.

Further reading
---------------

Core course references
~~~~~~~~~~~~~~~~~~~~~~

.. _ref-nvidia-cuda-aware:

**NVIDIA — An Introduction to CUDA-Aware MPI.**
Jiri Kraus, NVIDIA Technical Blog. Introduces host staging, direct device
pointers, Unified Virtual Addressing, transfer pipelining, GPUDirect P2P,
GPUDirect RDMA, and accelerated communication.
`Read the NVIDIA article <https://developer.nvidia.com/blog/introduction-cuda-aware-mpi/>`_.

.. _ref-intel-mpi-gpu:

**Intel — Intel MPI for GPU Clusters.**
Intel oneAPI GPU Optimization Guide, 2023.2. Covers host-managed and GPU-aware
execution models, GPU topology detection and pinning, scale-up and scale-out,
GPU-enabled Intel MPI Benchmarks, profiling, and runtime tuning.
`Read the Intel guide <https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2023-2/intel-mpi-for-gpu-clusters.html>`_.

.. _ref-epcc-mpi-gpus:

**EPCC — Using MPI with GPUs.**
ARCHER Introduction to GPU Programming course. Demonstrates rank-based device
selection and compares explicit host staging with direct GPU-aware MPI.
`Read the EPCC lesson <https://epcced.github.io/archer-gpu-course/section-5.02/>`_.

Additional references
~~~~~~~~~~~~~~~~~~~~~

* `Hands-On with Gadi: compute job basics <https://handson-with-gadi.readthedocs.io/en/latest/tutorial/jobs.html>`_
* `NCI Gadi user guide <https://opus.nci.org.au/display/Help/Gadi+User+Guide>`_
* `MPI Forum standards <https://www.mpi-forum.org/docs/>`_
* `Open MPI CUDA-aware FAQ <https://www.open-mpi.org/faq/?category=runcuda>`_
* `NVIDIA GPUDirect overview <https://developer.nvidia.com/gpudirect>`_
