FROM ubuntu:22.04
LABEL org.opencontainers.image.authors="Sadjad Fouladi <sfouladi@microsoft.com>"

RUN apt-get update -qq
RUN apt-get install -y python3 python3-pip openssh-client

COPY py/requirements.txt /app/
RUN pip3 install -r /app/requirements.txt

COPY py/ /app/
WORKDIR /app/

ENTRYPOINT ["python3", "coordinator.py"]
