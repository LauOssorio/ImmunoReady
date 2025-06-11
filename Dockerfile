FROM python:3.12.9-slim

COPY requirements.txt /requirements.txt
COPY immuno_ready /immuno_ready
COPY immunoready-b1f167dfb2c4.json /immuno_ready/gcp.json
COPY models /models

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc libffi-dev libssl-dev \
    && pip install --upgrade pip \
    && pip install -r /requirements.txt \
    && apt-get remove -y build-essential gcc \
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

CMD ["sh", "-c", "uvicorn immuno_ready.api.fast:app --host 0.0.0.0 --port $PORT"]
