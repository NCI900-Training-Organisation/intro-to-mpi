#!/bin/bash
set -euo pipefail
if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <example.pbs or saved job.txt path>" >&2
  exit 2
fi
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "$1" == *.pbs ]]; then
  record="$repo_dir/.pbs-jobs/${1##*/}.latest"
else
  record="$1"
fi
if [[ ! -f "$record" ]]; then
  echo "No submission record found: $record. Submit the job first." >&2
  exit 1
fi
{
  IFS= read -r job_id
  IFS= read -r output
} < "$record"
printf 'Job: %s\nOutput: %s\n' "$job_id" "$output"
qstat "$job_id" || echo "Status unavailable (the job may have left the queue). Inspect output for success or errors."
if [[ -f "$output" ]]; then
  cat -- "$output"
else
  echo "Output not available yet. Rerun this script after completion; do not resubmit."
fi
