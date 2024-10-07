# Use the ubuntu
FROM ubuntu:24.04

# Set the working directory to /app
WORKDIR /app

# install python and pip, and delete cache
RUN apt update && apt install -y python3 python3-pip python3.12-venv python3-dev libxml2-dev libxslt1-dev
RUN python3 -m venv /opt/venv

# Enable venv
ENV PATH="/opt/venv/bin:$PATH"
ADD . /app

# Install Python dependencies
RUN pip install -Ur requirements.txt

# Copy the current directory contents into the container at /app
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
