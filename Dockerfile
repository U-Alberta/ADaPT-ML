FROM python:3.8-slim-buster

WORKDIR /code
COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .