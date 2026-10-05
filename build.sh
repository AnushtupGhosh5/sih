#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
image_name="${IMAGE_NAME:-sih-dead-reckoning:latest}"
base_image="${BASE_IMAGE:-flipkart-grid:latest}"

if ! docker image inspect "$base_image" >/dev/null 2>&1; then
  echo "Base image '$base_image' is not available locally." >&2
  echo "Set BASE_IMAGE to one of your existing images and run this script again." >&2
  exit 1
fi

docker build \
  --build-arg "BASE_IMAGE=$base_image" \
  --tag "$image_name" \
  "$project_dir"

echo "Built $image_name from local base $base_image"
