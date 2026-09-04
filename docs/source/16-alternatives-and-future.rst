Alternatives and advanced designs
===================================

CUDA-aware MPI is one way to combine distributed communication with GPU
computation. The best choice depends on the communication graph, programming
model, portability target, and installed software.

NCCL and MPI
------------

NCCL is designed for high-performance GPU collectives such as all-reduce and
all-to-all patterns. MPI is broader: it provides process management, arbitrary
point-to-point communication, communicators, neighbourhood operations, and
many collective forms. Applications sometimes use MPI for control and
inter-node coordination and NCCL for GPU-heavy collectives. Compare the
available topology and failure behavior rather than assuming one replaces the
other.

NVSHMEM
-------

NVSHMEM provides a GPU-centric partitioned global address space and device
initiated communication. It can reduce CPU involvement for suitable algorithms
but introduces different synchronization, memory consistency, and deployment
requirements. It is not an MPI drop-in replacement.

PGAS, UCX, and lower-level transports
--------------------------------------

PGAS models, UCX-based communication, and vendor-specific libraries expose
other trade-offs between control, portability, and performance. MPI remains a
useful portability layer when an application needs standard process and
communicator semantics.

Multi-GPU process designs
-------------------------

A process may control one GPU, several GPUs, or share one GPU with other
processes. One process per GPU simplifies affinity and ownership. A multi-GPU
process can reduce MPI rank count but must manage peer access, streams, and
communication scheduling. Measure both designs on the target topology.

Resilience and long-running jobs
--------------------------------

Traditional MPI applications often abort the whole job after an unrecoverable
rank failure. Checkpointing, spare ranks, and fault-tolerant extensions require
additional design and support from the MPI implementation and scheduler. GPU
state, communication progress, and restart files must be considered together.

Future-facing features
----------------------

Relevant advanced areas include MPI sessions, partitioned communication,
GPU-initiated communication, CUDA graph integration, NIC offload, and
reproducible collectives. Treat these as implementation-specific topics and
validate them on the target Gadi software stack before teaching or deploying.
