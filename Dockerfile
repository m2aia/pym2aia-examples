FROM ubuntu:20.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3 \
        python3-pip \
        python3-venv \
        python3-setuptools \
        python3-wheel \
        wget \
        git \
        libglu1-mesa-dev \
        libtiff5-dev \
        libopenslide-dev \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*


RUN python3 -m pip install pip install https://github.com/m2aia/pym2aia/releases/download/v0.2.6/pym2aia-0.2.6-py3-none-linux_x86_64.whl
RUN python3 -m pip install wget matplotlib==3.5.0 seaborn seaborn_image tensorflow umap-learn torch torchvision tensorflow efficientnet_pytorch jupyter
 

VOLUME [ "/examples" ]

WORKDIR /examples
RUN mkdir -p results
ENTRYPOINT [ "sh", "/examples/run.sh" ]