# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

RUN mkdir /root/project

COPY emotions/ /root/project/emotions/
COPY requirements.txt /root/project/requirements.txt
COPY pyproject.toml /root/project/pyproject.toml
COPY config/ /root/project/config/
COPY data/ /root/project/data/

WORKDIR /root/project

RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir

ENV PYTHONPATH "${PYTHONPATH}:/root/project"

ENTRYPOINT ["sh", "entrypoint.sh"]