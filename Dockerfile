FROM registry.baidubce.com/paddlepaddle/paddle:2.4.2-gpu-cuda11.2-cudnn8.2-trt8.0

#-----------------------
# Copy working directory
#-----------------------
ARG WORKSPACE
COPY . ${WORKSPACE}

# -------------------------
# pip requirements
# -------------------------
RUN pip3 install -r ${WORKSPACE}/requirements.txt -i https://mirrors.aliyun.com/pypi/simple/

CMD ["python", "/app/app.py"]
EXPOSE 3333


