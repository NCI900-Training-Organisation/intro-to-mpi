# CUDA-aware MPI on Gadi

Reference and hands-on material for CUDA-aware MPI, developing a distributed
2-D Jacobi solver from ordinary MPI into a CUDA-aware, overlapped
implementation and covering memory, transport, topology, collectives,
debugging, portability, and performance.

## Core hands-on route

| Topic | Program |
|---|---|
| MPI/GPU process placement | `01-rank-device.cu` |
| Host-staged communication | `02-staged-pingpong.cu` |
| CUDA-aware point-to-point MPI | `03-device-pingpong.cu` |
| Blocking halo exchange | `04-jacobi-blocking.cu` |
| Nonblocking exchange and overlap | `05-jacobi-overlap.cu` |
| Benchmarking and correctness | both solvers |
| Message-size latency and bandwidth | `06-bandwidth.cu` |
| Device-buffer collectives | `07-device-collectives.cu` |
| Packed and datatype strided halos | `08-strided-halo.cu` |
| Managed-memory communication | `09-managed-memory.cu` |
| Device selection before `MPI_Init` | `10-preinit-device.cu` |
| CUDA stream ordering around MPI | `12-stream-aware-mpi.cu` |


## Quick start on Gadi

```console
$ git clone <this-repository-url>
$ cd intro-to-mpi
$ module purge
$ module load cuda/11.4.1 openmpi/4.1.5 cmake
$ cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70
$ cmake --build build --parallel
$ qsub job-script/01-rank-device.pbs
```

The PBS scripts only run executables already built in `build/bin`; they do not
invoke CMake or compile code. They target project `jxj900` and the `gpuvolta`
queue. See the [workshop documentation](docs/source/index.rst)
for the core route, extended reference chapters, commands, exercises, expected
output, and troubleshooting.

## Building the examples

Load CUDA, Open MPI, and CMake on Gadi, then configure an out-of-source build:

```console
$ module purge
$ module load cuda/11.4.1 openmpi/4.1.5 cmake
$ cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70
$ cmake --build build --parallel
```

Executables are written to `build/bin`. Build an individual example by naming
its target:

```console
$ cmake --build build --target 03-device-pingpong --parallel
```

Volta GPUs use architecture `70`. For another system, set
`CMAKE_CUDA_ARCHITECTURES` to the compute capability of its GPU. Build from
the repository root before submitting any PBS script; the scripts expect the
default `build/bin` output directory to be visible on the compute nodes.

## Repository layout

- `src/` — numbered CUDA/MPI source files
- `CMakeLists.txt` — CMake build definition for all examples
- `job-script/` — numbered PBS Pro scripts for Gadi
- `docs/source/` — Sphinx/Read the Docs workshop content
- `docs/requirements.txt` — pinned documentation dependencies

The documentation layout and visual configuration follow the NCI Training
Organisation's [`sphinx-book-theme` template](https://github.com/NCI900-Training-Organisation/template_repo/tree/sphinx-book-theme),
including its NCI logos, navigation controls, extension set, and dependency
versions.

Build the documentation locally with:

```console
$ python -m pip install -r docs/requirements.txt
$ sphinx-build -M html docs/source docs/_build
```

## Licence

BSD 3-Clause; see [LICENSE](LICENSE).

## Build a single example from a notebook

Run the notebook working-directory setup first. From an allocated compute-node
session, use the shared script with a source filename:

```bash
bash build-script/build-src.sh 04-jacobi-blocking.cu
```

The script also accepts `src/04-jacobi-blocking.cu`. It configures CMake and
builds only the matching target in `build/bin`. When environment modules are
available, it loads CUDA 12.9.0, Open MPI 4.1.5, and CMake, matching the runtime
PBS scripts. Otherwise these tools must already be on `PATH`. Compilation is
separate from PBS submission; do not compile on login nodes. If an existing
`build/` uses another toolchain, prepare a fresh build directory before running.

## Submit a PBS job and check status

From the repository root, after building the example:

```bash
qsub job-script/04-jacobi-blocking.pbs
qstat
```

Run `qsub` once to submit. Rerun `qstat` to check the queue without resubmitting.
`qstat` reports job status, not application output.
