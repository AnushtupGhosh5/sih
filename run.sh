#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image_name="${IMAGE_NAME:-sih-dead-reckoning:latest}"

if ! docker image inspect "$image_name" >/dev/null 2>&1; then
  echo "Image '$image_name' has not been built. Run ./build.sh once." >&2
  exit 1
fi

terminal_args=(-i)
if [[ -t 0 && -t 1 ]]; then
  terminal_args=(-it)
fi

gpu_args=()
if [[ "${USE_GPU:-auto}" != "0" ]] \
  && command -v nvidia-smi >/dev/null 2>&1 \
  && docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q 'nvidia'; then
  gpu_args=(--gpus all)
fi

# The bind mount keeps source, data, and generated reports on the host. It also
# means ordinary code/data changes never require another Docker build.
docker run --rm \
  "${terminal_args[@]}" \
  "${gpu_args[@]}" \
  --init \
  --user "$(id -u):$(id -g)" \
  --volume "$project_dir:/workspace" \
  --workdir /workspace \
  "$image_name" \
  "$@"
