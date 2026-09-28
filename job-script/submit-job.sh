#!/bin/bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <example.pbs>" >&2
  exit 2
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
script_name="${1##*/}"
if [[ "$script_name" != *.pbs || ! -f "$repo_dir/job-script/$script_name" ]]; then
  echo "Unknown PBS script: $1" >&2
  exit 2
fi
command -v qsub >/dev/null || { echo "qsub is unavailable" >&2; exit 1; }
cd "$repo_dir"
mkdir -p .pbs-jobs
run_dir=$(mktemp -d "$repo_dir/.pbs-jobs/${script_name%.pbs}.XXXXXX")
output="$run_dir/output.txt"
# Explicit output location avoids relying on the PBS job name or server suffix.
job_id=$(qsub -o "$output" -j oe "job-script/$script_name")
printf '%s\n%s\n' "$job_id" "$output" > "$run_dir/job.txt"
cp "$run_dir/job.txt" ".pbs-jobs/$script_name.latest"
printf 'Submitted %s\nOutput: %s\nRecord: %s/job.txt\n' "$job_id" "$output" "$run_dir"
qstat "$job_id" || echo "Status unavailable; check again using view-output.sh."
