Communication patterns
=======================

CUDA-aware MPI applies to more than a ping-pong. The right pattern depends on
the application's dependency graph and the shape of its data.

Point-to-point operations
-------------------------

Blocking ``MPI_Sendrecv`` is a useful halo-exchange baseline because it avoids
many ordering mistakes. ``MPI_Isend`` and ``MPI_Irecv`` allow independent work
to proceed while messages are active. Persistent communication requests can
reduce setup overhead for repeated, fixed-neighbour exchanges when supported
and beneficial.

Use distinct tags or carefully matched communicators for simultaneous message
classes. A tag scheme should identify the iteration and direction without
creating unnecessary tag management complexity.

Halo exchange designs
----------------------

For a row decomposition, each rank usually owns interior rows and stores one
ghost row per neighbour. A robust iteration is:

#. post receives into ghost rows;
#. post sends from owned boundary rows;
#. compute rows independent of incoming ghosts;
#. wait for communication;
#. compute boundary rows;
#. synchronize and swap arrays.

For wider stencils, store enough ghost layers or use a staged exchange. For
irregular domains, communicate explicit index lists or packed buffers.

Neighbour collectives
---------------------

``MPI_Neighbor_alltoallv`` and related neighbourhood collectives express a
fixed graph of communicating ranks. They can reduce application bookkeeping,
but device-buffer support, datatype support, and performance depend on the MPI
implementation. Compare them with explicit nonblocking point-to-point calls.

One-sided communication
-----------------------

MPI RMA operations such as ``MPI_Put`` and ``MPI_Get`` use windows and expose a
different synchronization model. Device-resident windows and GPU memory
ordering are not uniformly supported. Treat RMA as an advanced portability
case and verify the specific MPI, CUDA, and transport combination.

Partitioned and persistent communication
----------------------------------------

Partitioned communication can expose portions of a large message as they
become ready, which may help tiled GPU algorithms. Persistent requests can
amortize repeated request setup. Both features require careful buffer
ownership and are less portable than ordinary point-to-point operations.
