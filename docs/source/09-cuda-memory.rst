CUDA memory and MPI buffers
============================

CUDA-aware MPI depends on the kind of memory passed to MPI. The pointer must
remain valid for the entire MPI operation, and the application must establish
ordering between CUDA work and MPI access.

Memory classes
--------------

* ``cudaMalloc`` allocates device memory and is the normal device-buffer case.
* ``cudaMallocHost`` or ``cudaHostAlloc`` allocates page-locked host memory
  suitable for efficient DMA and asynchronous copies.
* ``malloc`` allocates pageable host memory. MPI may internally copy it to a
  pinned buffer before transfer.
* Unified memory allocated with ``cudaMallocManaged`` is portable at the CUDA
  API level, but device-pointer MPI support and migration behavior vary. Do not
  assume it has the same performance or support as ``cudaMalloc`` memory.
* Host-mapped and registered memory have implementation-specific restrictions.

Pinned memory is a limited resource. Allocate reusable staging buffers instead
of pinning every message, and release them when the communication phase ends.
Excessive pinning can reduce system performance.

Unified Virtual Addressing
--------------------------

UVA gives host and device allocations distinct address ranges in one virtual
address space on a node. CUDA-aware MPI can use the pointer value, often with
CUDA runtime queries, to identify a device buffer without adding a new MPI API.
UVA does not make a regular MPI build CUDA-aware and does not promise a direct
network transfer.

Buffer rules
------------

A send buffer must not be modified until the send request has completed. A
receive buffer must not be consumed until the receive has completed. A device
kernel must not write a buffer while MPI reads it, and MPI must not write a
buffer while a kernel consumes it. These rules apply even when a program often
appears to work due to timing.

Contiguous and noncontiguous data
---------------------------------

Start with contiguous device arrays. MPI derived datatypes can describe
strided layouts, but CUDA-aware support for device-resident derived datatypes
varies. A portable approach is to pack a halo into a contiguous device buffer
with a kernel, communicate it, and unpack it after completion. Packing adds
work, but may improve transport compatibility and memory access efficiency.

Memory registration and reuse
-----------------------------

High-performance transports may register GPU or host memory. Registration has
a cost, so repeated allocation and deregistration can distort short-message
benchmarks. Reuse allocations and communicate stable buffers when possible.
The MPI implementation may cache registrations, but applications should not
rely on undocumented cache behavior.

Hands-on: noncontiguous and managed buffers
-------------------------------------------

Submit ``jobs-scripts/08-strided-halo.pbs``. Its ``packed`` mode gathers a
matrix column into contiguous device memory with a CUDA kernel. Its
``datatype`` mode passes an ``MPI_Type_vector`` over device memory directly;
this is deliberately a capability test because support and performance vary.

Submit ``jobs-scripts/09-managed-memory.pbs`` to communicate a
``cudaMallocManaged`` allocation. It establishes kernel completion, performs
MPI, and prefetches the page to the CPU before host access. Compare this with
``cudaMalloc``: correctness does not imply equal performance because page
migration can add hidden transfers.
