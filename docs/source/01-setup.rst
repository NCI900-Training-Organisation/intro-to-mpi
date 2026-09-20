Setup and GPU placement
=======================

Prerequisites
-------------

Participants should know basic C/C++, CUDA kernels, and MPI point-to-point
calls. They need Gadi access, membership of the project ``vp91``.

Why combine MPI and CUDA?
-------------------------

CUDA accelerates computation within one node and MPI connects processes across
distributed-memory nodes. Combining them lets an application solve a problem
that is too large for one GPU, use several GPUs in parallel, or extend an
existing MPI application beyond one node. Each MPI process has a private
address space and is identified by a rank. Processes communicate through
point-to-point operations such as ``MPI_Send`` and ``MPI_Recv``; collective
operations such as ``MPI_Reduce`` combine values from several ranks.

The basic execution pattern is:

.. code-block:: c++

   MPI_Init(&argc, &argv);
   MPI_Comm_rank(MPI_COMM_WORLD, &rank);
   MPI_Comm_size(MPI_COMM_WORLD, &size);
   /* application work and MPI communication */
   MPI_Finalize();

The MPI compiler wrapper supplies the required headers and libraries, while
``mpirun`` or the PBS-launched equivalent starts one process for each rank.
This workshop uses MPI for distributed coordination and CUDA kernels for the
local grid computation.

Start in a clone on a filesystem visible to compute nodes, then submit:

.. code-block:: console

   $ qsub jobs-scripts/01-rank-device.pbs
   $ qstat -swx <job-id>
   $ less cuda-mpi-map.o<job-id>

The script requests two GPUs and two MPI ranks. ``01-rank-device.cu`` forms a
shared-memory communicator with ``MPI_Comm_split_type``. Its node-local rank
selects a CUDA device. Expected output maps local ranks 0 and 1 to different
GPUs on the same host. This follows the one-MPI-process-per-device pattern in
the :ref:`EPCC GPU course <ref-epcc-mpi-gpus>`, while using a node-local rank
rather than world rank so that the mapping remains valid across nodes.

Why local rank matters
----------------------

World rank is not a safe GPU index on multiple nodes: world rank 2 may be the
first process on a second node and must choose its local GPU 0. The program's
modulo fallback prevents an invalid device number, but oversubscribing ranks
onto a GPU is normally undesirable. Prefer one rank per requested GPU.

.. admonition:: Exercise (10 minutes)
   :class: exercise

   Change the script to request one GPU and run one rank. Predict and verify
   the local rank and device. Then explain why merely setting
   ``CUDA_VISIBLE_DEVICES`` globally is insufficient for several ranks.

PBS anatomy
-----------

``-P`` charges ``jxj900``; ``-q`` selects ``gpuvolta``; ``select`` describes
nodes, CPUs, GPUs, and memory; ``walltime`` is a hard limit; ``storage`` grants
compute-node access to project scratch; and ``-l wd`` starts in the submission
directory. ``mpirun --map-by ppr:N:node`` makes placement explicit.
