# FROM waggle/plugin-base:1.1.1-ml-torch1.9

# WORKDIR /app

# COPY requirements.txt .
# RUN pip3 install --no-cache-dir -r requirements.txt

# ADD https://github.com/giorgio-tran/fire/releases/download/yolov7/yolov7-fire.pt yolov7-fire.pt

# COPY . .

# ENTRYPOINT ["python3", "app.py"]


FROM waggle/plugin-base:1.1.1-ml-torch1.9

WORKDIR /app

RUN pip3 install --upgrade pip
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt

COPY utils/ /app/utils
COPY models/ /app/models
COPY app.py /app/

RUN wget -P /app/ https://github.com/giorgio-tran/fire/releases/download/yolov7/yolov7-fire.pt

ENTRYPOINT ["python3", "-u", "/app/app.py"]