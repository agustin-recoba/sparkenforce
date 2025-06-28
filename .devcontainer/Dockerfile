ARG PYTHON_VARIANT=3.10
FROM python:${PYTHON_VARIANT}

# Set the working directory
WORKDIR /workspace

# Copy the project files into the container
COPY . /workspace

# Install python package in editable mode
RUN pip3 --disable-pip-version-check --no-cache-dir \
   install -e .

# Install OpenJDK
RUN apt-get update && apt-get upgrade -y && \
    apt-get install software-properties-common -y && \
    apt-get install python3-launchpadlib -y && \
    add-apt-repository ppa:openjdk-r/ppa -y && \
    apt-get install default-jre -y && \
    apt-get clean;