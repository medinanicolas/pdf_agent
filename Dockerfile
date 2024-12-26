FROM python:3.12.4-alpine

RUN apk add gcc python3-dev musl-dev linux-headers

WORKDIR /agent

COPY ./requirements.txt /agent/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /agent/requirements.txt

COPY ./app /agent/app

CMD ["fastapi", "run", "app/main.py"]
