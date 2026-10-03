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

# Lets an orchestrator tell "serving" from "crashed in a restart loop".
# Implemented as a file rather than a one-liner so a failed probe prints one
# line instead of a traceback: during the 90s start period a cold boot fails
# every probe, and a traceback per probe buries anything real in the log.
#
# Copied to /algorithex-docker by the `COPY . /algorithex-docker` above. That
# stage is not the final one, whose WORKDIR is /home, hence the absolute path.
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python /algorithex-docker/docker_healthcheck.py

# Declare the port so `docker run -P` and orchestrators know what to map.
EXPOSE 9000 9001 9002
