
FROM ubuntu:18.04

WORKDIR /usr/src/app

ENV LANG="C.UTF-8" LC_ALL="C.UTF-8" PATH="/opt/venv/bin:$PATH" PIP_NO_CACHE_DIR="false"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv ffmpeg && \
    rm -rf /var/lib/apt/lists/*

RUN pip install pipenv

WORKDIR    /

ARG USERNAME
ARG PASSWORD

COPY Pipfile Pipfile
COPY Pipfile.lock Pipfile.lock
RUN pipenv install --deploy --system --sequential

ADD . /

CMD ["python", "main.py"]
