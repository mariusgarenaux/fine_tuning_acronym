FROM debian:12-slim

RUN chmod 777 /root
RUN apt-get update \
    && apt-get install -y acl \
    && apt-get install -y software-properties-common \
    && add-apt-repository -y "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) universe" \
    && apt-get update \
    && apt-get install zstd \
    && apt-get install -y curl ca-certificates \
    && apt-get install -y pciutils \
    && apt-get install -y git \
    && rm -rf /var/lib/apt/lists/*
RUN setfacl -d -m o::rwx /root

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# Pin / Install python version
RUN /root/.local/bin/uv python pin 3.12

# Install Google Collab libs (for bucket ?)
# Install GCSFuse
RUN apt-get update && apt-get install -y gnupg=2.2.40-1.1+deb12u2
RUN apt update && apt install -y libcurl4=7.88.1-10+deb12u14 curl=7.88.1-10+deb12u14 lsb-release=12.0-1 \
  && echo "deb https://packages.cloud.google.com/apt gcsfuse-$(lsb_release -c -s) main" | tee /etc/apt/sources.list.d/gcsfuse.list \
  && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
  && apt-get update && apt install -y fuse=2.9.9-6+b1 gcsfuse \
  && rm -rf /var/lib/apt/lists/*

# Install base packages for Jupyter Lab
# with dependencies for MCP Server and Collaboration
# WORKDIR /root
#RUN /root/.local/bin/uv venv
#RUN /bin/sh .venv/bin/activate

RUN https://github.com/mariusgarenaux/fine_tuning_acronym /root/fine_tuning_acronym
WORKDIR /root/fine_tuning_acronym
RUN /root/.local/bin/uv venv && /bin/sh .venv/bin/activate && /root/.local/bin/uv sync
RUN /root/.local/bin/uv pip install 'jupyterlab==4.4.1' 'jupyter-collaboration==4.0.2' 'jupyter-mcp-tools>=0.1.4' 'ipykernel' 'pycrdt'
RUN /root/.local/bin/uv pip install jupyterlab-miami-nights
RUN /root/.local/bin/uv pip install pydantic_ai_kernel
RUN /root/.local/bin/uv pip install ipywidgets

# Run Jupyter Lab
EXPOSE 8888
ENTRYPOINT ["/root/fine_tuning_acronym/.venv/bin/jupyter-lab", "--allow-root", "--IdentityProvider.token", "token", "--ServerApp.allow_remote_access", "True", "--ServerApp.base_url", "/notebook/", "--NotebookApp.token", "token"]
# CMD ["tail", "-f", "/dev/null"]