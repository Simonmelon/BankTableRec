#FROM nvidia/cuda:11.2.2-cudnn8-runtime-ubuntu20.04
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

ENV PYTHON_VERSION=3.8
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    curl \
    make \
    wget \
    ffmpeg \
    gcc \
    libjpeg-dev \
    libpng-dev \
    libavdevice-dev \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-dev \
    python3-tk \
    python${PYTHON_VERSION}-distutils \
    # set python
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python \
    && ln -sf /usr/bin/python${PYTHON_VERSION} /usr/bin/python3

#-----------------------
# Copy working directory
#-----------------------
ARG WORKSPACE
COPY . ${WORKSPACE}

# -------------------------
# Install core PIP packages.
# -------------------------
# Upgrade pip.
RUN python ${WORKSPACE}/deployment/docker/get-pip.py -i https://mirrors.aliyun.com/pypi/simple/

# -------------------------
# pip requirements
# -------------------------
RUN pip3 install -r ${WORKSPACE}/deployment/requirements_01.txt -i https://mirrors.aliyun.com/pypi/simple/
# cpu 版本
# RUN pip3 install paddlepaddle==2.4.2
# cuda 10.2
# RUN pip3  install paddlepaddle-gpu==2.4.2
# cuda 11.2
RUN pip3 install paddlepaddle-gpu==2.4.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
RUN pip3 install -r ${WORKSPACE}/deployment/requirements_02.txt -i https://mirrors.aliyun.com/pypi/simple/

ENV SERVICE_HOST="0.0.0.0"
ENV SERVICE_PORT=3333
ENV SERVICE_MODE="gpu"
ENV SERVICE_MODEL_NUM=1

#CMD ["python", "/app/bankstatement_ocr_flask/flask/bankstatement_ocr_app.py"]
CMD python /app/bankstatement_ocr_flask/deployment/flask/bankstatement_ocr_app.py --host ${SERVICE_HOST} --port ${SERVICE_PORT} --mode ${SERVICE_MODE} --model_num ${SERVICE_MODEL_NUM}

