FROM python:3.5.3

WORKDIR /app/
COPY requirements.txt /app/
RUN pip install -r ./requirements.txt
COPY static /app/static
COPY templates /app/templates
ADD ./main.py /app/

ADD ./func.py /app/
EXPOSE 8080
ENTRYPOINT python /app/main.py