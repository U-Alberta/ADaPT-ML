FROM python:3.8-slim-buster

WORKDIR /code

# Install essentials
RUN apt-get update \
    && apt-get install -y \
        build-essential \
        git \
        wait-for-it

RUN apt update \
    && apt install -y \
        apt-transport-https \
        ca-certificates \
        curl \
        wget \
        gnupg-agent \
        software-properties-common \
        libyaml-cpp-dev \
        libyaml-dev

# Install Miniconda and create conda env
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.4-Linux-x86_64.sh -O ~/anaconda.sh \
    && /bin/bash ~/anaconda.sh -b -p /opt/conda \
    && rm ~/anaconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh \
    && echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
ENV PATH /opt/conda/bin:$PATH
COPY ./conda.yaml .
COPY ./setup.cfg .
COPY ./setup.py .
COPY ./label ./label
RUN conda env create -f conda.yaml
RUN echo "conda activate dp_env" >> ~/.bashrc

CMD [ "/bin/bash" ]
