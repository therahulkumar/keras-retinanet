FROM python:3

WORKDIR /home

COPY app.py .

RUN apt-get update -y && \
    git clone https://github.com/fizyr/keras-retinanet.git && \
    pip install numpy && \
    cd keras-retinanet && \
    pip install . && \
    pip install tensorflow==1.13.1 && \
    pip install Flask && \
    export FLASK_APP=app.py


CMD ["bash python -m", "flask run"]




