ARG PYTHON_VARIANT=3.10
ARG UV_VER=0.5.24
ARG DEBIAN_VER=bookworm
ARG WORKSPACE_NAME_=workspace
ARG PROJECT_NAME_=sparkenforce

# Install python
FROM ghcr.io/astral-sh/uv:$UV_VER AS uv

FROM mcr.microsoft.com/vscode/devcontainers/base:$DEBIAN_VER

# Install/Update linux packages; install common dev tools like: git, process tools, ...
# hadolint ignore=DL3008
RUN apt-get update \
    && apt-get install -y --no-install-recommends\
    procps \
    build-essential \
    curl \
    swig \
    wget \
    # To reduce the image size, it is recommended refresh the package cache as follows.
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copies files or directories from the uv stage into the final stage,
# and ensures that the ownership of the copied files is adjusted to the user and group in the final image,
# and making its functionality or binaries available in the final container.
COPY --from=uv --chown=vscode: /uv /uvx /bin/

# Set the working directory
WORKDIR /${WORKSPACE_NAME_}


# Install OpenJDK
RUN apt-get update && apt-get upgrade -y && \
    apt-get install software-properties-common -y && \
    apt-get install python3-launchpadlib -y && \
    add-apt-repository ppa:openjdk-r/ppa -y && \
    apt-get install default-jre -y && \
    apt-get clean;