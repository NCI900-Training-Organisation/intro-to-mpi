Debugging and portability
==========================

Start with a correct host-staged implementation, then substitute device
buffers. This isolates CUDA kernel errors, MPI placement errors, and
CUDA-aware transport errors.

Layered validation
------------------

#. Run one rank on one GPU and validate the kernel result.
#. Run two ranks with host staging and compare against a CPU reference.
#. Run direct device-buffer point-to-point operations.
#. Move from one node to multiple nodes.
#. Add nonblocking operations and overlap only after correctness is stable.
#. Test collectives and derived datatypes independently.

Useful diagnostics
------------------

.. code-block:: console

   $ module list
   $ mpicxx --showme
   $ ompi_info --parsable -l 9 | grep -i cuda
   $ nvidia-smi
   $ CUDA_LAUNCH_BLOCKING=1 ./application

Use the MPI implementation's supported diagnostic options rather than copying
MCA or environment settings from a different version. Capture module versions,
job placement, GPU model, driver, CUDA version, and MPI configuration with each
benchmark result.

Initialisation order is another implementation difference. Some MPI libraries
create accelerator state in ``MPI_Init`` and need the device selected first;
others initialise lazily. ``10-preinit-device.cu`` recognises common launcher
local-rank variables, selects CUDA before ``MPI_Init``, then verifies the choice
with ``MPI_Comm_split_type``. Run ``jobs-scripts/10-preinit-device.pbs``.

Intel GPU clusters pose the same execution-model questions but use different
APIs: SYCL USM or OpenMP target device pointers, Level Zero topology, Intel MPI
GPU pinning, and ``I_MPI_OFFLOAD``. Those Intel controls do not belong in
Gadi's CUDA/Open MPI scripts. On an Intel system, inspect MPI debug output
because runtime affinity controls can override one another.

Typical failures
----------------

**Invalid device pointer:** the MPI library, selected transport, or operation
does not support the supplied memory. Confirm that the staged version works.

**Intermittent wrong answers:** a kernel, asynchronous copy, or nonblocking
MPI request is still using a buffer when another operation modifies it.

**Deadlock:** ranks disagree about peers, tags, counts, communicators, or
collective participation. Check rank-local logs and simplify to blocking
``MPI_Sendrecv``.

**Works on one node but not two:** the inter-node transport or GPUDirect RDMA
path is unavailable, misconfigured, or unsupported for the allocation.

**No speedup:** staging may already be hidden, messages may be too small, the
kernel may be too short for overlap, or the network path may dominate. Measure
rather than infer from API syntax.

Portability matrix
------------------

Record support across the combinations that matter to the application:

.. list-table::
   :header-rows: 1

   * - Operation
     - Host buffer
     - Device buffer
     - Device datatype
   * - Point-to-point
     - usually supported
     - test implementation
     - test implementation
   * - Reduction
     - usually supported
     - test operation
     - not applicable
   * - RMA
     - test implementation
     - test implementation
     - test implementation

Treat this as a test plan, not a standards guarantee. MPI standard semantics do
not imply CUDA-memory support for every implementation or transport.
