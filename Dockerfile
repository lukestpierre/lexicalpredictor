FROM python:latest

WORKDIR /usr/src/app

COPY requirements.txt ./

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN export FLASK_APP=app

RUN flask run -p 80
