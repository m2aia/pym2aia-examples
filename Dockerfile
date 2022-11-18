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


# m2aia/m2aia:latest-build containes the latest M2aia installer
# check https://m2aia.github.io/m2aia for download the packacge manually
COPY --from=m2aia/m2aia:latest-package /opt/packages/m2aia.tar.gz /opt/packages/m2aia.tar.gz

# we extract all files to this location
RUN mkdir /opt/m2aia
RUN tar -xvf /opt/packages/m2aia.tar.gz -C /opt/m2aia --strip-components=1

# promote the required library path
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/m2aia/bin
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/m2aia/bin/MitkCore

RUN python3 -m pip install git+https://github.com/m2aia/pym2aia jupyter wget
RUN python3 -m pip install matplotlib seaborn


VOLUME [ "/examples" ]

WORKDIR /examples
RUN mkdir -p results
ENTRYPOINT [ "sh", "/examples/run.sh" ]