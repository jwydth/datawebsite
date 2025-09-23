# Use a Miniconda base image (lightweight Python + Conda)
FROM continuumio/miniconda3:latest

# Set working directory
WORKDIR /app

# Copy environment file first (for better Docker caching)
COPY requirements.yml .

# Create a conda environment named "flaskenv" from requirements.yml
RUN conda env create -f requirements.yml -n flaskenv

# Make sure conda environment is used by default
SHELL ["conda", "run", "-n", "flaskenv", "/bin/bash", "-c"]

# Copy application code
COPY . .

# Expose Flask default port
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# Run Flask inside the conda env
RUN conda run --no-capture-output -n flaskenv python models/gen_model.py
CMD conda run --no-capture-output -n flaskenv python app.py
