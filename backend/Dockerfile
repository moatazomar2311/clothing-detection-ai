FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies for TFLite + Pillow
RUN apt-get update && apt-get install -y \
    libgl1 \
    libx11-6 \
    libxext6 \
    libxrender1 \
    libjpeg62-turbo \
    zlib1g \
    && apt-get clean

# Copy requirements
COPY requirements.txt ${LAMBDA_TASK_ROOT}/

# Install Python libraries
RUN pip install --no-cache-dir -r ${LAMBDA_TASK_ROOT}/requirements.txt

# Copy model
COPY new_model.tflite ${LAMBDA_TASK_ROOT}/

# Copy Lambda code
COPY lambda_function.py ${LAMBDA_TASK_ROOT}/

CMD ["lambda_function.lambda_handler"]

