FROM python:3.8-slim-buster

WORKDIR /code

COPY requirements.txt .
COPY setup.py .
COPY setup.cfg .
COPY ./ls ./ls
RUN pip install -r requirements.txt

CMD [ "/bin/bash" ]
