CUDA-aware MPI capability and transport diagnostics
========================================================

This chapter shows how to investigate which communication path carries a
CUDA device buffer. Start with ``03-device-pingpong`` and then repeat the
investigation for the Jacobi solvers. See :doc:`11-gpudirect-topology` for the
hardware background and :doc:`15-performance-methodology` for timing guidance.

What are you trying to identify?
-------------------------------------

CUDA-aware MPI accepts supported device pointers. It does not promise a
particular transport. Distinguish the MPI backend, the selected transfer
protocol, and the physical connection.

.. list-table:: Possible payload paths
   :header-rows: 1
   :widths: 22 43 35

   * - Path
     - Data movement
     - Typical setting
   * - GPUDirect P2P
     - GPU memory to GPU memory over NVLink or PCIe
     - Peer-accessible GPUs within one node
   * - GPUDirect RDMA
     - GPU memory to network adapter, across the network, to remote GPU memory
     - Compatible GPUs and network adapters on different nodes
   * - Internal staging
     - GPU memory through intermediate host buffers
     - A fallback or a protocol selected for a particular message size

Direct payload movement avoids host-memory staging; the CPU can still initiate
and manage the MPI call. CUDA IPC lets separate local processes share access
to GPU allocations. NVLink is a hardware interconnect, not an MPI backend.
The NVIDIA references below explain the GPUDirect mechanisms and requirements.

.. important::

   The ``device-direct`` text printed by ``03-device-pingpong`` describes the
   application's use of a device pointer. It is not a measurement or proof of
   GPUDirect. Likewise, a successful transfer proves neither RDMA nor P2P use.

Run a diagnostic compute job
---------------------------------

Build the examples separately using the CMake instructions in ``README.md``
and the site's permitted build environment. Do not compile or execute GPU
programs on a login node. Save this diagnostic script as
``transport-check.pbs`` in the repository root. It runs prebuilt executables
and keeps the repository's current Gadi module settings in one place.

.. code-block:: bash

   #!/bin/bash
   #PBS -P vp91
   #PBS -q gpuvolta
   #PBS -l ncpus=24
   #PBS -l ngpus=2
   #PBS -l mem=8gb
   #PBS -l jobfs=1GB
   #PBS -l walltime=00:10:00
   #PBS -l storage=scratch/vp91
   #PBS -l wd
   #PBS -N mpi-transport
   #PBS -j oe


   set -eu
   module purge
   module load cuda/11.4.1 openmpi/4.1.5


   mpirun --version
   ompi_info --parsable --all > "ompi-info.${PBS_JOBID}.txt"
   nvidia-smi -L
   nvidia-smi topo -m
   if command -v ucx_info >/dev/null 2>&1; then
     ucx_info -v
     ucx_info -d
   fi

   mpirun -np 2 --map-by ppr:2:node build/bin/01-rank-device

   mpirun -np 2 --map-by ppr:2:node --tag-output \
     --mca pml_base_verbose 100 \
     --mca btl_base_verbose 30 \
     --mca opal_cuda_verbose 10 \
     --mca mpi_common_cuda_verbose 20 \
     -x UCX_LOG_LEVEL=info \
     build/bin/03-device-pingpong 4194304

Submit from the repository root and inspect the merged output after completion:

.. code-block:: console

   $ qsub transport-check.pbs
   $ qstat JOB_ID
   $ less mpi-transport.oJOB_NUMBER

Replace the placeholders with the returned job identifier and the actual output
filename. The rank-device output should show two ranks on the same hostname,
with different GPU assignments. For the ping-pong example, rank 0 should report
``received 1``. Its argument is a count of floats: ``4194304`` represents 16 MiB
per buffer on this platform. The example checks only a sample, not every value.

The topology matrix describes connectivity and affinity. Entries such as
``NV#`` identify NVLink connections; PCIe and CPU-related labels are explained
in the matrix legend. Connectivity shows a possible route, not traffic actually
observed on that route.

Identify the MPI backend first
-----------------------------------

Inspect the PML selection messages, distinguishing a component being considered
from one being selected. Open MPI may select ``ucx`` or ``ob1``; do not assume
that loading Open MPI means UCX was selected.

* With ``ucx``, UCX diagnostics describe communication protocols and transports.
* With ``ob1``, inspect BTL selection and CUDA diagnostics. Components such as
  ``smcuda`` or ``openib`` depend on the installed build. UCX logging does not
  describe traffic handled by another backend.

Search the saved ``ompi-info`` file for ``mpi_built_with_cuda_support`` to check
Open MPI's build-level CUDA support. This is capability information, not proof
that a particular operation or backend can use every CUDA memory type.
The Open MPI 4.x CUDA-aware FAQ documents the CUDA verbosity parameters used
above. Their output depends on the build and selected backend.

Do the first run without forcing ``--mca pml ucx`` or restricting ``UCX_TLS``.
Forcing a backend changes the experiment; it does not reveal the original
selection. If testing a forced configuration later, record that change.

Check the installed Open MPI toolchain
-------------------------------------------

Before running a GPU test, inspect the MPI wrapper and build configuration.
These metadata commands can run on the login node; they do not execute a GPU
application. For the newer candidate toolchain discussed here, use:

.. code-block:: bash

   module purge
   module load cuda/12.9.0 openmpi/5.0.8
   module list
   nvcc --version
   mpirun --version
   mpicxx --showme
   ompi_info --parsable --all | grep mpi_built_with_cuda_support
   ompi_info --param pml ucx --level 9

The observed Gadi output included:

.. code-block:: text

   g++ -Wl,--push-state -Wl,--as-needed -lmpi -Wl,--pop-state
   mca:mpi:base:param:mpi_built_with_cuda_support:value:true
   MCA pml: ucx (MCA v2.1.0, API v2.1.0, Component v5.0.8)

Interpret these results as follows:

* ``g++`` is the host C++ compiler invoked by the MPI wrapper. The project's
  CUDA source files are compiled with ``nvcc`` and linked to MPI through CMake.
* ``value:true`` confirms CUDA support was built into Open MPI. The surrounding
  synonym and enumerator lines describe the parameter, not additional tests.
* ``Component v5.0.8`` confirms that the UCX PML component is present in this
  Open MPI installation. It does not establish runtime selection or the UCX
  library's own version.

The reported ``pml_ucx_tls`` value included
``rc_verbs,ud_verbs,rc_mlx5,dc_mlx5,ud_mlx5,cuda_ipc,rocm_ipc``.
This is a list of criteria for selecting the UCX component, not an inventory
of working transports or a trace of an actual transfer. Similarly,
``pml_ucx_devices=mlx*`` influences component priority when matching devices
are available; it does not prove that a GPU payload used a Mellanox adapter.

Validate the newer toolchain in a compute job
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The output above makes Open MPI 5.0.8 a candidate for these examples, but does
not validate CUDA 12.9 driver compatibility or GPUDirect operation. Rebuild in a
fresh build directory using the chosen toolchain and the repository's CMake
instructions, in an authorised build environment. Keep the Volta architecture
setting ``CMAKE_CUDA_ARCHITECTURES=70``. Do not compile GPU programs on the login
node or add compilation commands to the runtime PBS script.

For an Open MPI 5.0.8 diagnostic run, retain the PBS resource directives from
the script above. Replace its module loading with
``module load cuda/12.9.0 openmpi/5.0.8`` and use the following execution block.
This example assumes the fresh build directory is ``build-ompi5``:

.. code-block:: bash

   nvcc --version
   nvidia-smi

   mpirun -np 2 --map-by ppr:2:node \
     build-ompi5/bin/01-rank-device

   mpirun -np 2 --map-by ppr:2:node --tag-output \
     --mca pml_base_verbose 100 \
     --mca pml_ucx_verbose 10 \
     -x UCX_LOG_LEVEL=info \
     build-ompi5/bin/03-device-pingpong 4194304

Submit the script with ``qsub``. Confirm two different GPU assignments,
``received 1`` from rank 0, and the backend selected in the runtime log.
``nvidia-smi`` records the compute-node driver; its displayed CUDA version is
not the version of the toolkit selected by the module. Record ``nvcc --version``
as well. Successful execution still does not prove GPUDirect use: continue
with the transport evidence checks below.

Use identical toolchain modules when building and running. The repository's
existing PBS scripts still load CUDA 11.4.1 and Open MPI 4.1.5; they must not be
used unchanged to validate the new build. This diagnostic variant does not
change the repository's default toolchain.

Interpret UCX evidence carefully
-------------------------------------

``ucx_info -d`` inventories available transports; ``UCX_LOG_LEVEL=info`` adds
runtime information. A missing ``ucx_info`` executable does not by itself prove
that Open MPI lacks UCX support. Consult the module provider if its tools are
not on ``PATH``.

.. list-table:: Common UCX names
   :header-rows: 1
   :widths: 25 75

   * - Name
     - Interpretation
   * - ``cuda_ipc``
     - Local interprocess GPU access; investigate this for a same-node P2P path.
   * - ``cuda_copy``
     - CUDA copying support, including staging protocols.
   * - ``gdr_copy``
     - GDRCopy access between CPU and GPU memory; not network GPUDirect RDMA.
   * - ``rc_mlx5``, ``rc_verbs``, ``dc_mlx5``
     - RDMA-capable network transports. Their names alone do not establish that
       the payload came directly from GPU memory.

An endpoint configuration can contain multiple lanes. Seeing ``cuda_copy`` and
an RDMA transport together does not establish whether the transfer was staged.
Record the buffer memory type, message size, selected protocol, and lane used.
The UCX FAQ describes transport logging and protocol diagnostics.

For installations supporting protocol v2, inspect the available settings:

.. code-block:: bash

   ucx_info -f > "ucx-config.${PBS_JOBID}.txt"

If protocol v2 is active and ``UCX_PROTO_INFO`` is supported, add
``-x UCX_PROTO_INFO=y`` to the diagnostic ``mpirun`` command. It can report
protocol choices by memory type and message-size range. Older versions may
lack this facility. Do not enable a different protocol engine merely to claim
that its output describes the original run.

Testing inter-node GPUDirect RDMA
--------------------------------------

The script above tests two ranks on one node. To investigate network transfers,
obtain a Gadi allocation spanning at least two GPU nodes under the current queue
rules. Keep 12 CPU cores per requested GPU and use the site's documented flat
PBS resource style. A two-GPU request alone does not guarantee two nodes.
Consult the Gadi queue documentation for a suitable multi-node allocation.

Within that allocation, use the following placement for both the rank-device
check and the diagnostic ping-pong command:

.. code-block:: bash

   mpirun -np 2 --map-by ppr:1:node build/bin/01-rank-device

Replace ``--map-by ppr:2:node`` with ``--map-by ppr:1:node`` in the diagnostic
command as well. Verify different hostnames before interpreting the run as an
inter-node test. Capture GPU/NIC topology on each participating node:

.. code-block:: bash

   mpirun -np 2 --map-by ppr:1:node --tag-output nvidia-smi topo -m

GPUDirect RDMA needs compatible hardware, GPU-memory registration support,
drivers, and a suitable transport. Depending on the software generation,
registration may involve peer-memory modules or DMA-BUF. Ask the administrators
about the installed configuration; the presence of a driver module is not
proof that an MPI message used it.

Strong evidence is a selected protocol for CUDA memory that transfers directly
through the network adapter. A generic ``zero-copy`` label without its memory
type and protocol context is insufficient. If the installed logs cannot
establish the path, report it as unconfirmed and request implementation-specific
tracing or profiling assistance. Timing alone cannot prove GPUDirect RDMA.

Correctness and performance checks
---------------------------------------

Keep CUDA producer completion before MPI and MPI request completion before
buffer reuse. Preserve these rules when applying diagnostics to
``04-jacobi-blocking`` and ``05-jacobi-overlap``. See
:doc:`10-synchronization` for the ownership and ordering requirements.

Remove verbose logging for performance measurements. Use ``06-bandwidth`` to
compare message sizes and repeat runs with the same placement. Its host-buffer
column measures MPI on a host buffer after staging; it is not an end-to-end
measurement including both GPU copies. Neither that comparison nor a single
ping-pong sample establishes the transport by itself.

Managed memory and derived datatypes can select different paths or have more
limited support than contiguous ``cudaMalloc`` buffers. Establish the simple
device-buffer case first, then investigate ``08-strided-halo`` and
``09-managed-memory`` separately. For collectives, also account for the selected
collective implementation and algorithm; point-to-point evidence need not
apply to ``MPI_Allreduce``.

Exercises and evidence record
----------------------------------

#. Run the same-node diagnostic. Record rank placement, GPU assignment, MPI and
   UCX versions, topology, selected backend, and relevant protocol messages.
#. Repeat with float counts ``256`` and ``4194304``. Explain any protocol change
   without assuming the largest message uses the same path as the smallest.
#. In an appropriate two-node allocation, repeat with one rank per node.
   Separate evidence of RDMA capability from evidence of direct GPU payload
   transfer.
#. Apply the logging to the blocking and overlapped Jacobi solvers. Explain why
   nonblocking MPI does not by itself guarantee a different transport or overlap.

For each result, state the scope: operation, allocation type, message size,
placement, runtime overrides, observed path, and any uncertainty. Keep the full
job output alongside the conclusion so the evidence can be reviewed.

References
---------------

* `Open MPI 4.x CUDA-aware FAQ <https://www.open-mpi.org/faq/?category=runcuda>`_
* `UCX FAQ and diagnostic settings <https://github.com/openucx/ucx/blob/master/docs/source/faq.md>`_
* `NVIDIA GPUDirect overview <https://developer.nvidia.com/gpudirect>`_
* `NVIDIA GPUDirect RDMA requirements <https://docs.nvidia.com/cuda/archive/11.8.0/gpudirect-rdma/index.html>`_
* `NVIDIA topology and RDMA benchmarking <https://developer.nvidia.com/blog/benchmarking-gpudirect-rdma-on-modern-server-platforms/>`_
* `Gadi compute job documentation <https://handson-with-gadi.readthedocs.io/en/latest/tutorial/jobs.html>`_

The Open MPI reference targets version 4.x, matching the repository's module.
The UCX reference tracks current development; use the installed version's
configuration output to determine which settings are available.
