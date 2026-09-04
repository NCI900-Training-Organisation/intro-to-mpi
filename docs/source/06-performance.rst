Performance experiment
======================

Submit ``jobs-scripts/04-jacobi-blocking.pbs`` and
``jobs-scripts/05-jacobi-overlap.pbs`` to compare the two solver versions.
For a size sweep, change the ``nx``, ``ny``, and iteration arguments in both
scripts identically and copy the results into a table:

.. list-table::
   :header-rows: 1

   * - Grid
     - Blocking (s)
     - Overlap (s)
     - Speedup
     - Gcell updates/s
   * - 2048²
     -
     -
     -
     -
   * - 4096²
     -
     -
     -
     -
   * - 8192²
     -
     -
     -
     -

Use ``speedup = T_blocking / T_overlap`` and
``Gupdates/s = nx*ny*iterations/time/1e9``. Repeat runs, retain node placement
and module versions, add untimed warm-up iterations for publishable results,
and report variability. A fair scaling study holds either global problem size
(strong scaling) or work per GPU (weak scaling) constant.

What controls performance?
--------------------------

Message size, GPU/CPU/PCIe/NVLink/NIC topology, intra- versus inter-node
transport, network contention, MPI progress, kernel duration, synchronisation,
and GPU affinity all matter. Use ``nvidia-smi topo -m`` only on an allocated
GPU node. Profile with tools approved on Gadi, but first establish correctness
and a simple timing baseline.

.. admonition:: Final challenge (15 minutes)
   :class: exercise

   Add a global residual: compute ``max(abs(new-old))`` on each GPU and combine
   the scalar with ``MPI_Allreduce``. Stop at a tolerance. Discuss why reducing
   one host scalar may be preferable to relying on device-buffer collective
   support for this operation.
