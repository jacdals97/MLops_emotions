# Base image
FROM python:3.11-slim

RUN apt update && \
    apt install --no-install-recommends -y build-essential gcc && \
    apt clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
COPY pyproject.toml pyproject.toml
COPY emotions/ emotions/
COPY config/ config/

WORKDIR /
RUN pip install -r requirements.txt --no-cache-dir
RUN pip install . --no-deps --no-cache-dir
COPY .dvc .dvc
COPY data.dvc data.dvc
RUN dvc config core.no_scm true
RUN dvc pull

ENTRYPOINT ["python", "-u", "emotions/train_model.py"]