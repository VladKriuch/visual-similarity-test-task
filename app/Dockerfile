FROM pytorch/pytorch:2.6.0-cuda11.8-cudnn9-runtime

WORKDIR /app

# Install Python and required packages
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    python3.11-dev \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container
COPY . ./

# Create virtual environment and install dependencies
RUN python3.11 -m venv venv && pip install --upgrade pip \
    && . venv/bin/activate

RUN . venv/bin/activate && pip install --no-cache-dir -r requirements.txt
EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
# Run the script with the specified arguments
CMD ["/bin/bash", "-c", "source venv/bin/activate && python populate_db.py && streamlit run app.py --server.port=8501 --server.address=0.0.0.0"]