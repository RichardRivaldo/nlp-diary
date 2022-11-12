FROM python:alpine

EXPOSE 80

# Install waitress & falcon
RUN pip install waitress falcon

# Add app
COPY . /app
WORKDIR /app

CMD ["bash ./run.sh"]