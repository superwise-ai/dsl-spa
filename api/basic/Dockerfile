# Use Python 3.10-slim as a parent image
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Install dependencies
RUN apt-get update -y && apt-get install -y --no-install-recommends gnupg2 wget && \
    wget --quiet -O - "https://packages.microsoft.com/keys/microsoft.asc" | gpg --dearmor --yes -o /usr/share/keyrings/microsoft.gpg && \
    echo "deb [arch=amd64,arm64,armhf signed-by=/usr/share/keyrings/microsoft.gpg] https://packages.microsoft.com/debian/12/prod $(grep -oP 'VERSION_CODENAME=\K\w+' /etc/os-release) main" | tee /etc/apt/sources.list.d/mssql-release.list && \
    apt-get update -y && \
    ACCEPT_EULA=Y apt-get install -y --no-install-recommends libodbc2 msodbcsql17 mssql-tools \
    wkhtmltopdf && \
    apt-get clean && \
    rm -rf /var/lib/apt/*
COPY requirements.txt /app/
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copy the current directory contents into the container at /app
COPY *.py ./
COPY connectors.json ./
COPY .env ./

# Add mssql-tools to PATH
ENV PATH="${PATH}:/opt/mssql-tools/bin"

# Make port 80 available to the world outside this container
EXPOSE 80

# Run uvicorn server when the container launches
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]

