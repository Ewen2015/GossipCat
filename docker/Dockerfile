# Use an official Python runtime as a parent image
# FROM python:3.6.3
FROM ewenwang/gossipcat:packages

MAINTAINER Ewen Wang <Wang.Enqun@outlook.com>

# Install any needed packages specified in requirements.txt
# RUN pip install -r requirements.txt

# Lauch Jupyter when the container launches
ADD run.sh /
RUN chmod +x /run.sh

CMD ["/run.sh"]