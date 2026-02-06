

FROM python:3 

LABEL version="1.0"

FROM python:3

WORKDIR /usr/src/app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r  requirements.txt

COPY sk_forecast_rms.py runner ./

VOLUME ["/usr/src/app/data"]


ENTRYPOINT bash /usr/src/app/runner
