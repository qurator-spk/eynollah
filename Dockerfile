ARG DOCKER_BASE_IMAGE
FROM $DOCKER_BASE_IMAGE

ARG VCS_REF
ARG BUILD_DATE
LABEL \
    maintainer="https://ocr-d.de/kontakt" \
    org.label-schema.vcs-ref=$VCS_REF \
    org.label-schema.vcs-url="https://github.com/qurator-spk/eynollah" \
    org.label-schema.build-date=$BUILD_DATE

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONIOENCODING=utf8
ENV XDG_DATA_HOME=/usr/local/share

WORKDIR /build-eynollah
COPY src/ ./src
COPY pyproject.toml .
COPY requirements.txt .
COPY README.md .
COPY Makefile .
RUN apt-get install -y --no-install-recommends g++
RUN make install

WORKDIR /data
VOLUME /data
