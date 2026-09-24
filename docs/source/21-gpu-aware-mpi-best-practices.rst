GPU-aware MPI best practices
============================

This checklist summarizes the practical rules used throughout the repository.
It is specific to MPI communication involving CUDA memory, not a general MPI
or CUDA guide.

Buffer ownership and ordering
-----------------------------

* Complete the producer CUDA kernel before MPI reads a send buffer.
* Do not modify a send buffer until its nonblocking MPI request completes.
* Do not launch a consumer kernel until the receive or RMA operation completes.
* Use CUDA events and explicit stream dependencies when work is not on the
  default stream.
* Keep the communication buffer alive for the entire MPI request or RMA epoch.

Support and transport
---------------------

* Verify CUDA-aware support in the loaded MPI build, not only at compile time.
* Test point-to-point, collectives, RMA, derived datatypes, and managed memory
  separately; support is not necessarily uniform across operations.
* Treat a successful device-pointer call as API support, not proof of GPUDirect
  RDMA or zero-copy transport.
* Record the CUDA, MPI, UCX or OFI, driver, GPU, NIC, and topology versions for
  reproducible measurements.

Rank, GPU, and NIC placement
----------------------------

* Map ranks using node-local placement rather than assuming global rank equals
  GPU number.
* Check GPU-to-GPU and GPU-to-NIC locality with topology tools on allocated
  compute nodes.
* Avoid accidental GPU oversubscription and measure the effect of NUMA and PCIe
  locality.

Communication patterns
----------------------

* Start with contiguous device buffers and explicit point-to-point operations.
* Use nonblocking operations only when buffer ownership and MPI progress are
  clear.
* Use packing kernels for strided device data when derived-datatype support or
  performance is uncertain.
* Compare explicit halo exchange with neighborhood collectives, persistent
  requests, and RMA only after establishing a correct baseline.
* Treat GPU-initiated communication as a separate programming model; standard
  MPI calls are normally launched by the CPU.

Progress and overlap
--------------------

* Do not assume MPI progresses while a GPU kernel occupies the device.
* Measure communication and kernel timelines independently before claiming
  overlap.
* Test whether the MPI library needs CPU polling or an asynchronous progress
  thread for the selected transport.
* Use warm-up iterations and synchronize at measurement boundaries.

Memory choices
--------------

* Prefer ordinary device allocations for the clearest CUDA-aware MPI baseline.
* Test managed memory separately; page migration and registration can dominate
  communication time.
* Reuse buffers when possible to reduce allocation and memory-registration
  overhead.
* Keep host staging buffers pinned when explicit staging is required.

Correctness experiments
-----------------------

For every new GPU-aware MPI pattern, test at least:

* one rank and multiple ranks;
* one node and multiple nodes;
* small and large messages;
* default and non-default CUDA streams;
* first-use and warmed-up iterations; and
* a result compared with a CPU or host-staged reference.

Report the MPI implementation and transport configuration with performance
results. A result that is correct on one cluster does not establish portability
to another CUDA-aware MPI stack.