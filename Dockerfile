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

# RUN curl -L -o /app/yolov7-fire.pt "https://www.dropbox.com/scl/fi/g843zo26x2u27fm09irt7/best.pt?rlkey=8ijy7pxme9cpgfpxqk9swwyvo&st=9ulyplea&dl=1"

RUN curl -L -o /app/model.pt "https://web.lcrc.anl.gov/public/waggle/models/vehicletracking/yolov7.pt"

ENTRYPOINT ["python3", "-u", "/app/app.py"]