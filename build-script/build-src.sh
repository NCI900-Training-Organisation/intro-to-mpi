#!/bin/bash
set -euo pipefail

# Build one repository example from its CUDA source filename.
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# Read the explicit target list used by CMake's add_executable loop.
targets=()
while IFS= read -r name; do
  targets+=("$name")
done < <(awk '
  /^[[:space:]]*set\(WORKSHOP_TARGETS[[:space:]]*$/ { in_targets = 1; next }
  in_targets {
    last = /\)/
    sub(/#.*/, "")
    gsub(/[()]/, "")
    for (i = 1; i <= NF; i++) print $i
    if (last) exit
  }
' "$repo_dir/CMakeLists.txt")

usage()
{
  echo "Usage: $0 <example.cu or src/example.cu>"
  echo "       $0 --list"
}

if [[ $# -ne 1 ]]; then
  usage >&2
  exit 2
fi
case "$1" in
  -h|--help) usage; exit 0 ;;
  --list) printf '%s.cu\n' "${targets[@]}"; exit 0 ;;
esac

source_name="${1##*/}"
target="${source_name%.cu}"
if [[ ! "$target" =~ ^[0-9]+-[a-z0-9-]+$ || "$source_name" != *.cu || ! -f "$repo_dir/src/$source_name" ]]; then
  echo "Unknown CUDA source: $1" >&2
  exit 2
fi
registered=false
for name in "${targets[@]}"; do
  if [[ "$name" == "$target" ]]; then
    registered=true
  fi
  if [[ ! -f "$repo_dir/src/$name.cu" ]]; then
    echo "CMake target $name has no source file: src/$name.cu" >&2
    exit 2
  fi
done
if [[ "$registered" != true ]]; then
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
