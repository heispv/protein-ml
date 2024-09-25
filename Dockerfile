# Use the official Python 3.9 slim image as base
FROM python:3.9-slim-buster

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    libfreetype6-dev \
    libxft-dev \
    zlib1g-dev \
    libbz2-dev \
    && rm -rf /var/lib/apt/lists/*

# Install MMseqs2
RUN wget https://mmseqs.com/latest/mmseqs-linux-sse41.tar.gz -O /tmp/mmseqs-linux-sse41.tar.gz && \
    tar xzf /tmp/mmseqs-linux-sse41.tar.gz -C /tmp && \
    mv /tmp/mmseqs/bin/mmseqs /usr/local/bin/ && \
    rm -rf /tmp/mmseqs-linux-sse41.tar.gz /tmp/mmseqs

# Copy requirements.txt and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the project code into /app
COPY . /app
WORKDIR /app

# Set the default command to bash
CMD ["bash"]
