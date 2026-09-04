GPUDirect and hardware topology
=================================

CUDA-aware MPI is an interface capability. The communication path selected at
runtime depends on the MPI build, transport modules, driver versions, GPU
placement, network adapter, message size, and node topology.

GPUDirect families
------------------

**GPUDirect P2P** accelerates transfers between GPUs in one node when the
hardware and peer-access path permit it. GPUs may communicate through PCIe,
NVLink, or a host-mediated path depending on the topology.

**GPUDirect RDMA** allows a compatible network adapter to transfer data directly
between GPU memory and the network for inter-node communication. It requires a
compatible GPU, NIC, driver, CUDA, MPI transport, and system configuration.

**GPUDirect accelerated communication** removes unnecessary copies between
intermediate CUDA and network-fabric buffers in supported paths.

These three communication paths and their relationship to CUDA-aware MPI are
introduced in :ref:`NVIDIA's article <ref-nvidia-cuda-aware>`.

Without these paths, a CUDA-aware MPI implementation may accept a device
pointer and internally stage through pinned host memory. That is still a valid
CUDA-aware API behavior, but it has different performance characteristics.

Topology and affinity
---------------------

Inspect topology only from an allocated GPU node:

.. code-block:: console

   $ nvidia-smi topo -m
   $ nvidia-smi -L

Important relationships include GPU-to-CPU affinity, PCIe root complexes,
NVLink connectivity, NIC locality, NUMA placement, and whether ranks sharing a
GPU are intentional. A correct rank-to-GPU mapping can outperform a faster
kernel running with a remote or contended network path.

One rank per GPU is a simple default. More advanced designs may use multiple
ranks per GPU, one process controlling several GPUs, or a separate communication
process. These choices change memory ownership, synchronization, and resource
contention and should be measured rather than assumed.

Transport decision tree
-----------------------

For each transfer, ask:

#. Are the source and destination buffers device allocations supported by the
   loaded MPI library?
#. Are the ranks on one node or different nodes?
#. Is a GPU peer path available for the local transfer?
#. Is the NIC and software stack configured for GPU RDMA?
#. If not, which internal staging path is used?
#. Does the measured result justify the added complexity?

The API call alone cannot answer these questions. Use MPI diagnostics,
implementation documentation, topology tools, and controlled benchmarks.
