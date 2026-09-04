Performance methodology
========================

CUDA-aware MPI performance is a system property. A fair experiment controls
the application, allocation, placement, software environment, and measurement
region.

Communication metrics
---------------------

For a message of size $n$ bytes, latency is the time for a small transfer and
bandwidth is approximately $n/t$. Report whether the time includes kernel
launches, CUDA synchronization, memory copies, barriers, and MPI startup.
Separate warm-up iterations from measured iterations.

``06-bandwidth.cu`` sweeps power-of-two sizes from one byte to 64 MiB with
warm-ups and repeated measurements. Its pinned-host column times MPI only;
explicit device-to-host and host-to-device copies must be included when
comparing complete staged application paths. ``jobs-scripts/06-bandwidth.pbs``
runs two GPUs within a node; ``jobs-scripts/06-bandwidth-scaleout.pbs``
requests two full Volta nodes to exercise the scale-out path.

A useful application metric for the Jacobi example is:

.. math::

   \text{cell updates/s} = \frac{n_x n_y N_{iterations}}{T}

Use the maximum rank time for a distributed iteration because the slowest rank
sets the application's progress. Report mean, minimum, maximum, and spread
across repeated runs when possible.

Benchmark matrix
----------------

Compare at least:

* host-staged versus direct device-buffer MPI;
* one-node versus multi-node placement;
* blocking versus nonblocking exchange;
* small, medium, and large messages;
* pinned versus pageable host staging;
* one rank per GPU versus any oversubscribed configuration;
* direct contiguous buffers versus packed halos.

Keep the MPI module, CUDA version, GPU model, node allocation, rank mapping,
CPU binding, message size, iteration count, and compiler flags fixed while
comparing one variable.

Overlap analysis
----------------

If communication takes $T_c$ and independent computation takes $T_k$, ideal
overlap approaches ``max(T_c, T_k)`` instead of ``T_c + T_k``. Real systems
lose time to launch overheads, synchronization, contention, insufficient MPI
progress, and shared resource pressure. A shorter kernel can reduce overlap
benefit even when the communication path is unchanged.

Scaling studies
---------------

Strong scaling holds global problem size constant while increasing GPUs;
parallel efficiency often falls as communication and synchronization dominate.
Weak scaling grows the problem with GPU count so work per GPU stays constant;
it reveals communication growth and placement effects. State which definition
is used and include the baseline.

Reproducibility
---------------

Run multiple repetitions, use equivalent allocations, record node placement,
and avoid timing compilation or queue wait time. Use CUDA events for isolated
GPU work and ``MPI_Wtime`` with an appropriate distributed timing design for
end-to-end application behavior. Do not put a barrier inside every measured
phase unless it represents a real application dependency.
