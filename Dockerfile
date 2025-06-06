FROM python:3.12.9-slim

COPY requirements.txt /requirements.txt
COPY immuno_ready /immuno_ready

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libffi-dev libssl-dev
RUN pip install --upgrade pip
RUN pip install -r /requirements.txt

CMD uvicorn immuno_ready.api.fast:app --host 0.0.0.0 --port $PORT
