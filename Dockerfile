FROM python:3.8-slim-buster

WORKDIR /code

RUN apt update \
    && apt install -y build-essential \
    && apt install -y git

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "/bin/bash" ]