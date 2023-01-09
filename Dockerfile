# Note: we nvidia/cuda such that this same image can be used for DL projects.
# If you don't need CUDA, you should a base image that is smaller like python:3.8-slim
FROM nvidia/cuda:11.7.1-base-ubuntu20.04

# Gather the required apt-get images
RUN apt-get -y update \
    && apt-get install -y software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get -y update \
    && add-apt-repository universe
# Install python 3.8
RUN apt-get -y update \
    && apt-get -y install python3.8 \
    && apt-get -y install python3-pip \
    && apt-get -y install python3.8-distutils

# Install poetry
RUN pip3 install poetry==1.3.1
ENV PYTHONIOENCODING=utf8
