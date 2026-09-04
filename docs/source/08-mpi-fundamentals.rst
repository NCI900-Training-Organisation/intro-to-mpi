MPI foundations
===============

MPI is a message-passing standard for processes with separate address spaces.
A rank is one process in a communicator. ``MPI_COMM_WORLD`` contains all
launched processes; application communicators can restrict communication to a
subset of ranks.

Core concepts
-------------

* ``MPI_Init`` and ``MPI_Finalize`` delimit MPI use.
* ``MPI_Comm_rank`` returns the calling process's rank.
* ``MPI_Comm_size`` returns the number of processes in a communicator.
* A message is identified by communicator, source, destination, tag, datatype,
  and count.
* Point-to-point operations include blocking and nonblocking sends and
  receives.
* Collective operations involve every rank in a communicator and include
  barriers, broadcasts, reductions, gathers, scatters, and all-to-all calls.

A message's datatype and count describe the logical payload, while the buffer
address describes where the payload lives. CUDA-aware MPI extends the normal
MPI buffer convention so that a supported implementation can accept a device
pointer.

Process launch and placement
----------------------------

The launcher starts one process per rank and establishes the environment used
by the MPI library. On a cluster, placement determines which ranks share a
node, socket, GPU, or network adapter. Use explicit placement options and
verify the result from inside the job; launcher syntax differs between MPI
implementations.

.. code-block:: console

   $ mpicxx --showme
   $ mpirun --map-by ppr:2:node ./application

The second command is an Open MPI example. On Gadi, use the supplied PBS
scripts and the launcher configuration supported by the loaded module.

Communicators and topology
--------------------------

``MPI_Comm_split_type`` can create a communicator containing ranks on the same
node. The rank within that communicator is useful for selecting a local GPU.
Cartesian communicators created with ``MPI_Cart_create`` can represent a
stencil's logical topology and provide neighbour discovery through
``MPI_Cart_shift``.

Common beginner errors
----------------------

Do not use world rank directly as a GPU index on multiple nodes. Match send and
receive counts, datatypes, tags, and peers. Every rank must participate in a
collective in the same communicator, and a blocking receive must have a
matching send. Always check MPI return codes in teaching and development code.
