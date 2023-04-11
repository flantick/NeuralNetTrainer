# you can choose your cuda version
FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime  
WORKDIR /usr/src/app
RUN apt-get update \
 && apt-get install -y libgl1-mesa-glx libgtk2.0-0 libsm6 libxext6 \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

CMD ["python3"]
