# Use an official Python runtime as a parent image
FROM python:3.6.3

MAINTAINER Ewen Wang <Wang.Enqun@outlook.com>

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Define environment variable
ENV NAME World

# Run app.py when the container launches
CMD ["python", "app.py"]