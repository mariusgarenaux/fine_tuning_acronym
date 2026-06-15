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
    && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*
    RUN setfacl -d -m o::rwx /root
    
RUN apt-get update \
    && apt-get install nano \
    && rm -rf /var/lib/apt/lists/*


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

# a .venv for agent helper
WORKDIR /root
RUN /root/.local/bin/uv venv && /bin/sh .venv/bin/activate
RUN /root/.local/bin/uv pip install pydantic_ai_kernel
RUN /root/.local/bin/uv pip install ipywidgets

# on .venv for the exercise and Jupyter Lab (allow for simple kernel choice)
RUN git clone https://github.com/mariusgarenaux/fine_tuning_acronym /root/fine_tuning_acronym
WORKDIR /root/fine_tuning_acronym
RUN git checkout formation-continue
RUN /root/.local/bin/uv venv && /bin/sh .venv/bin/activate && /root/.local/bin/uv sync


# RUN /root/.local/bin/uv pip install 'jupyterlab==4.4.1' 'jupyter-collaboration==4.0.2' 'jupyter-mcp-tools>=0.1.4' 'ipykernel' 'pycrdt' 'jupyterlab-miami-nights'
RUN /root/.local/bin/uv pip install 'jupyterlab==4.4.1' 'jupyterlab-miami-nights'

# add agent to the kernel that runs jupyter lab
RUN git clone https://github.com/mariusgarenaux/io_kernel /root/io_kernel
RUN /root/.local/bin/uv build /root/io_kernel && /root/.local/bin/uv pip install /root/io_kernel
RUN /root/fine_tuning_acronym/.venv/bin/jupyter kernelspec install /root/io_kernel/agent/data_kernelspec/share/jupyter/kernels/io
RUN rm -rf /root/io_kernel
RUN mkdir /root/.jupyter
RUN cp /root/fine_tuning_acronym/jupyter_io_config.yaml /root/.jupyter/jupyter_io_config.yaml

RUN /root/.local/bin/uv cache clean

# Run Jupyter Lab
EXPOSE 8888
ENTRYPOINT ["/bin/sh", "/root/fine_tuning_acronym/setup_datalab.sh"]
# CMD ["tail", "-f", "/dev/null"]