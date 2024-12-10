ARG HF_TOKEN

FROM python:3.11.11-slim-bullseye

ENV HF_HOME=/root/.cache/huggingface
ENV HF_TOKEN=${HF_TOKEN}

RUN mkdir -p /home/workspace/smol-course
WORKDIR /home/workspace/smol-course

COPY . .

RUN pip install --no-cache-dir -U -r requirements.txt

RUN python 0_start_with_docker/download.py

RUN echo "alias ll='ls -la'" >> ~/.bashrc