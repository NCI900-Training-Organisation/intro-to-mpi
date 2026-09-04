CUDA-aware MPI on Gadi
======================

This documentation is a comprehensive guide to using MPI with CUDA GPUs. It
builds one application—a distributed 2-D Jacobi heat solver—in stages, then
extends the core material with memory models, transport paths, collectives,
topology, debugging, portability, and performance methodology.

.. note::

   The examples target NCI Gadi, project ``jxj900``, queue ``gpuvolta``, and
   NVIDIA Volta GPUs. Module versions and queue policy change: run
   ``module avail cuda openmpi`` and check NCI documentation before teaching.

Learning outcomes
-----------------

By the end you can explain MPI ranks and communicators, map one MPI rank to
each GPU, pass device pointers to MPI, describe UVA and GPUDirect paths,
manage CUDA/MPI synchronisation, implement direct halo exchanges, overlap
interior computation with communication, validate results, and reason about
topology, collectives, derived datatypes, managed memory, portability, and
performance. The first six chapters form the core route; the remaining
chapters and examples are an extended course rather than a four-hour limit.

Core hands-on route
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
   * - :doc:`06-performance`
     - Benchmark and interpretation

.. toctree::
   :maxdepth: 2
   :caption: Core workshop
   :numbered:

   01-setup
   02-staging
   03-cuda-aware
   04-jacobi
   05-overlap
   06-performance

.. toctree::
   :maxdepth: 2
   :caption: Extended topics and reference
   :numbered:

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
