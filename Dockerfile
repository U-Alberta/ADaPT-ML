FROM python:3.8-slim-buster
WORKDIR /code

ENV FLASK_APP serving/app.py
ENV FLASK_RUN_HOST 0.0.0.0

RUN apt update \
    && apt install -y build-essential
#    && apt-get install -y python3-sklearn python3-sklearn-lib python3-sklearn-doc

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
#    && pip install pip-tools \
#    && pip-compile requirements.in > requirements.txt \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 5000
CMD ["flask", "run"]
