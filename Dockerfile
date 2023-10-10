# Use an official Python runtime as a parent image
FROM python:3.8.7-slim

# Set the working directory in the container
WORKDIR /contegris_app

# Copy the current directory contents into the container at app
ADD . /contegris_app

RUN pip install --upgrade pip
# Install any needed packages specified in requirements.txt
RUN pip install  -r requirements.txt

# # Make port 80 available to the world outside this container
# EXPOSE 80

# # Define environment variable
# ENV app=app.py

CMD python app.py
