#!/bin/bash
set -euo pipefail

# Build one repository example from its CUDA source filename.
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <example.cu or src/example.cu>" >&2
  exit 2
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source_name="${1##*/}"
target="${source_name%.cu}"
if [[ ! "$target" =~ ^[0-9]+-[a-z0-9-]+$ || "$source_name" != *.cu || ! -f "$repo_dir/src/$source_name" ]]; then
  echo "Unknown CUDA source: $1" >&2
  exit 2
fi
if ! grep -Eq "^[[:space:]]*${target}[[:space:]]*\\)?$" "$repo_dir/CMakeLists.txt"; then
  echo "No CMake example target is registered for $source_name" >&2
  exit 2
fi
case "$(hostname)" in
  *login*) echo "Build on an allocated compute node, not a login node." >&2; exit 1 ;;
esac

# Match the toolchain used by the runtime PBS scripts.
if type module >/dev/null 2>&1; then
  module purge
  module load cuda/12.9.0 openmpi/4.1.5 cmake
fi
for tool in cmake nvcc mpicxx; do
  if ! command -v "$tool" >/dev/null 2>&1; then
    echo "Required build tool unavailable: $tool" >&2
    exit 1
  fi
done

cmake -S "$repo_dir" -B "$repo_dir/build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70
cmake --build "$repo_dir/build" --target "$target" --parallel 2
echo "Built $repo_dir/build/bin/$target"
