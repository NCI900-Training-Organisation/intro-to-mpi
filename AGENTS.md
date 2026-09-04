# Repository instructions

## Project scope

This repository contains comprehensive teaching and reference material for
CUDA-aware MPI. Do not limit the material to a four-hour workshop. Cover the
full topic where useful, including:

- MPI fundamentals, ranks, communicators, point-to-point operations, and
	collectives.
- CUDA memory types, pinned memory, Unified Virtual Addressing, and buffer
	ownership.
- Host-staged and direct device-buffer communication.
- CUDA/MPI synchronization, CUDA streams, nonblocking operations, and MPI
	progress.
- GPUDirect P2P, GPUDirect RDMA, internal staging, and hardware topology.
- Halo exchanges, derived datatypes, neighbourhood communication, reductions,
	and other collective patterns.
- Correctness, debugging, portability, benchmarking, scaling, and performance
	methodology.
- Alternatives and related technologies such as NCCL, NVSHMEM, and lower-level
	communication libraries.

Use the distributed 2-D Jacobi heat solver as the central example. Develop it
from a host-staged baseline to direct CUDA-aware MPI and finally to an
overlapped nonblocking implementation.

## Repository layout

- `src/` contains numbered CUDA/MPI source files.
- `CMakeLists.txt` defines the CMake build for all examples.
- `jobs-scripts/` contains numbered PBS Pro scripts for running prebuilt
	examples on Gadi.
- `doc/` contains numbered Sphinx and Read the Docs documentation.

Keep source files, job scripts, and documentation numbered in a consistent
learning order. Update the documentation navigation whenever a numbered
chapter is added or renamed.

## Gadi job scripts

Target NCI Gadi with project `jxj900` and queue `gpuvolta`. Follow the Gadi job
documentation:

https://handson-with-gadi.readthedocs.io/en/latest/tutorial/jobs.html

Use the flat PBS resource style:

```bash
#PBS -P jxj900
#PBS -q gpuvolta
#PBS -l ncpus=24
#PBS -l ngpus=2
#PBS -l mem=8gb
#PBS -l jobfs=1GB
#PBS -l walltime=00:10:00
#PBS -l storage=scratch/jxj900
#PBS -l wd
```

Allocate 12 CPU cores per GPU. Adjust `ncpus` and `ngpus` together when a job
uses a different number of GPUs. Load the CUDA and Open MPI modules required by
the applications, currently:

```bash
module purge
module load cuda/11.4.1 openmpi/4.1.5
```

Separate each job script into three visual sections with blank lines:

1. PBS directives
2. Shell setup and module loading
3. Application execution

Do not put build commands such as `make`, `cmake`, `nvcc`, or compiler wrappers
in job scripts. Build the examples separately with CMake as documented in
`README.md`. Job scripts must only load the runtime environment and run the
matching prebuilt executable from `build/bin`.

Do not compile or execute GPU programs on a login node. Use `qsub` and verify
rank placement, GPU assignment, and output from the compute job.

## Source formatting

- Put every function's opening brace on the following line.
- Put `if`, `else`, `for`, and `while` opening braces on the same line as the
	control statement, separated by a space.
- Use consistent, readable indentation and avoid unrelated reformatting.
- Separate logical code blocks with two blank lines where practical. Examples
	of logical blocks include initialization, validation, allocation,
	communication, computation, timing, reporting, and cleanup.
- Preserve the existing public interfaces, example numbering, and behavior
	unless a change is required by the task.

## Documentation

Use clear reStructuredText compatible with Sphinx and the Read the Docs theme.
Document both the API-level behavior and the implementation-dependent behavior
of CUDA-aware MPI. Do not claim that passing a device pointer guarantees
GPUDirect RDMA; explain that the MPI implementation may select direct P2P,
RDMA, internal staging, or another transport.

Include commands, expected behavior, exercises, correctness guidance, and
portability warnings for new technical topics. Keep Gadi-specific module names
and resource settings easy to update as the platform changes.

## Validation

Before completing a change:

- Run `bash -n jobs-scripts/*.pbs` or an equivalent loop for PBS scripts.
- Run `git diff --check`.
- Configure with CMake to verify build wiring when source or `CMakeLists.txt`
	changes are involved, when CUDA and MPI development tools are available.
- Build the documentation when the Sphinx dependencies are available.
- Do not submit jobs or require CUDA hardware for local validation. Report
	clearly when `nvcc`, CUDA headers, Gadi, or the Read the Docs theme are not
	available locally.
