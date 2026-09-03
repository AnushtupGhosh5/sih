# Reuse the existing local ML image. Override at build time with:
#   BASE_IMAGE=another-image:tag ./build.sh
ARG BASE_IMAGE=flipkart-grid:latest
FROM ${BASE_IMAGE}

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace

# Project code and data are bind-mounted by run.sh, so changing Python files or
# datasets does not require rebuilding this image.
ENTRYPOINT ["/bin/bash", "/workspace/runScript.sh"]
