ARG TEST_BUILD=0
FROM python:3.11-slim-bookworm AS algorithex_basic_env
ENV PYTHONUNBUFFERED 1

RUN apt-get update \
    && apt-get -y install git build-essential libssl-dev \
    && apt-get clean \
    && pip install --upgrade pip

RUN pip3 install Cython numpy

# Prepare environment
RUN mkdir /algorithex-docker
WORKDIR /algorithex-docker

# Install dependencies
COPY requirements.txt /algorithex-docker
RUN pip3 install -r requirements.txt

# Build
COPY . /algorithex-docker
RUN pip3 install -e .

FROM algorithex_basic_env AS algorithex_with_test_0
WORKDIR /home

FROM algorithex_basic_env AS algorithex_with_test_1
RUN pip3 install codecov pytest-cov
ENTRYPOINT pytest --cov=./ # && codecov

FROM algorithex_with_test_${TEST_BUILD} AS algorithex_final
