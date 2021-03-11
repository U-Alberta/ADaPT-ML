FROM python:3.8-slim-buster

WORKDIR /code

# Install essentials
RUN apt-get update \
    && apt-get install -y build-essential

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY ./ls ./ls

CMD [ "/bin/bash" ]
