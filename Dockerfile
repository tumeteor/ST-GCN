FROM nvidia/cuda:10.1-cudnn7-devel

WORKDIR /usr/src/app

ENV LANG="C.UTF-8" LC_ALL="C.UTF-8" PATH="/opt/venv/bin:$PATH" PIP_NO_CACHE_DIR="false" CFLAGS="-mavx2" CXXFLAGS="-mavx2"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3.6 python3-pip python3-venv python3-dev \
    wget make g++ ffmpeg python3-dev libblas-dev liblapack-dev swig libsnappy-dev \
    libjpeg-turbo8-dev zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && \
    python3 -m pip install pip==19.2.2 pip-tools==4.0.0 pipenv

WORKDIR    /

ARG USERNAME
ARG PASSWORD

COPY requirements.txt requirements.txt
#COPY Pipfile Pipfile
#COPY Pipfile.lock Pipfile.lock
#RUN python3 -m pipenv install --verbose --deploy --system --sequential
RUN python3 -m pip install -r requirements.txt --extra-index-url https://$USERNAME:$PASSWORD@nexus.mobilityservices.io/repository/pypi/simple

ADD . /


