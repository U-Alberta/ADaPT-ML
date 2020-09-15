FROM python:3.8-slim-buster

#ARG USER_ID=${USER_ID}
#ARG GROUP_ID=${GROUP_ID}
#
#RUN addgroup --gid $GROUP_ID user
#RUN adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID user
#USER user

WORKDIR /code

# Install essentials
RUN apt update \
    && apt install -y \
        build-essential \
        git \
        wait-for-it

# install essentials for Docker
RUN apt-get update \
    && apt-get install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        gnupg-agent \
        software-properties-common \
        libyaml-cpp-dev \
        libyaml-dev

# Install Docker
RUN curl -fsSL https://download.docker.com/linux/debian/gpg | apt-key add -
RUN apt-key fingerprint 0EBFCD88
RUN add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian $(lsb_release -cs) stable"
RUN apt-get update \
    && apt-get install -y \
        docker-ce \
        docker-ce-cli \
        containerd.io

COPY ./requirements.txt .
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install  postgreSQL
#RUN apt-key adv --keyserver hkp://p80.pool.sks-keyservers.net:80 --recv-keys B97B0AFCAA1A47F044F244A07FCC7D46ACCC4CF8
#RUN echo "deb http://apt.postgresql.org/pub/repos/apt/ precise-pgdg main" > /etc/apt/sources.list.d/pgdg.list
#RUN apt-get update \
#    && apt-get install -y \
#    postgresql \
#    postgresql-client \
#    postgresql-contrib
#
#USER postgres
#RUN /etc/init.d/postgresql start \
#    && psql --command "CREATE USER mlflow_user WITH SUPERUSER PASSWORD 'mlflow';" \
#    && createdb -O mlflow_user /mlflow_label/mlflow_label_db
#USER root

COPY ./label ./label
COPY ./MLproject .
COPY ./conda.yaml .
CMD [ "/bin/bash" ]
