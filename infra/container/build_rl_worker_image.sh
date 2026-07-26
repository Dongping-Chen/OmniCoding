#!/usr/bin/env bash
# Build the CPU-side coding-agent runtime and export it as an Enroot squashfs
# that Slurm/Pyxis can mount independently for every rollout job step.
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
image_tag="${ROLLOUT_DOCKER_IMAGE:-omnicoding-rl-worker:local}"
output_image="${ROLLOUT_CONTAINER_IMAGE:-${repo_root}/runtime/omnicoding-rl-worker.sqsh}"

mkdir -p "$(dirname -- "${output_image}")"

docker build \
  --file "${repo_root}/infra/container/rl_worker.Dockerfile" \
  --tag "${image_tag}" \
  "${repo_root}"

if [[ -e "${output_image}" ]]; then
  echo "refusing to overwrite existing image: ${output_image}" >&2
  echo "set ROLLOUT_CONTAINER_IMAGE to a new versioned path" >&2
  exit 2
fi

enroot import --output "${output_image}" "dockerd://${image_tag}"
echo "wrote ${output_image}"
