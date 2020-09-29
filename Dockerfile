FROM python:3.8-slim-buster

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
        wget \
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

# Install Miniconda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/anaconda.sh \
    && /bin/bash ~/anaconda.sh -b -p /opt/conda \
    && rm ~/anaconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc

# Create conda env
ENV PATH /opt/conda/bin:$PATH

COPY ./conda.yaml .
RUN conda env create -f conda.yaml
RUN echo "conda activate cecn_env" >> ~/.bashrc

# Finish up
COPY ./modeling ./modeling
COPY ./MLproject .
CMD [ "/bin/bash" ]
