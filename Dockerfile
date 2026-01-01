FROM python:3.11-slim


WORKDIR /app


# Install dependencies
RUN pip install --no-cache-dir \
    fastapi==0.128.0 \
    uvicorn==0.40.0 \
    pillow==12.0.0 \ 
    python-multipart==0.0.21

RUN pip install --no-cache-dir \
    torch==2.9.1 \
    torchvision==0.24.1 \
    --extra-index-url https://download.pytorch.org/whl/cpu


# Copy app and model weights
COPY . . 


# Expose port the app listens on
EXPOSE 8000


# Start FastAPI app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]