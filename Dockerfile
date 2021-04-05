FROM python:3.6

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

WORKDIR /app
ADD . /app

# Install dependencies
RUN pip install -r requirements.txt

# Expose port
ENV PORT 8080

# Run the application:
CMD ["gunicorn", "app:app", "--config=config.py"]

# command we use to build docker image:

# docker image build -t <REPOSITORY> .
# docker images
# docker ps
# docker run -p 5000:5000 -d <REPOSITORY>
# docker login dockerfitbit.azurecr.io
# docker push dockerfitbit.azurecr.io/mlfitbit:latest
# docker stop <containerID>
# docker system prune
