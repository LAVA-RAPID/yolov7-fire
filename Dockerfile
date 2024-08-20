# FROM waggle/plugin-base:1.1.1-ml-torch1.9

# WORKDIR /app

# COPY requirements.txt .
# RUN pip3 install --no-cache-dir -r requirements.txt

# ADD https://github.com/giorgio-tran/fire/releases/download/yolov7/yolov7-fire.pt yolov7-fire.pt

# COPY . .

# ENTRYPOINT ["python3", "app.py"]


FROM waggle/plugin-base:1.1.1-ml-torch1.9

RUN apt-get update \
  && apt-get install -y wget\
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip3 install --upgrade pip
COPY requirements.txt /app/
RUN pip3 install --no-cache-dir --upgrade -r /app/requirements.txt

COPY utils/ /app/utils
COPY models/ /app/models
COPY app.py /app/

# RUN curl -L -o /app/yolov7-fire.pt "https://www.dropbox.com/scl/fi/g843zo26x2u27fm09irt7/best.pt?rlkey=8ijy7pxme9cpgfpxqk9swwyvo&st=9ulyplea&dl=1"
RUN wget -O /app/yolov7_fire_aa "https://www.dropbox.com/scl/fi/l0bqp1uow6vv3xax27ppe/best_part_aa?rlkey=87sjbsgj6bymnj6v5o7s56gmv&st=d8vrweq8&dl=1"
RUN wget -O /app/yolov7_fire_ab "https://www.dropbox.com/scl/fi/ky9m9ljbufm8i7vna02uh/best_part_ab?rlkey=q47gajcwkg62ut5kf9ijxwo6a&st=itybx00t&dl=1"
RUN wget -O /app/yolov7_fire_ac "https://www.dropbox.com/scl/fi/ymifqc988hv3cqgfm02q8/best_part_ac?rlkey=338utubz8zu6sh7rntbm6egcj&st=cgq1y9v9&dl=1"
RUN wget -O /app/yolov7_fire_ad "https://www.dropbox.com/scl/fi/1alvqmbi1siwbbvm6ipf7/best_part_ad?rlkey=sywh3zo3sqop7aim3m0wm1e62&st=rnspf1fw&dl=1"
RUN wget -O /app/yolov7_fire_ae "https://www.dropbox.com/scl/fi/gosvs1xdqouagevss2u7j/best_part_ae?rlkey=tal4dpyyu1hlf8fzy0gaxxseb&st=ql2h4yhn&dl=1"
RUN wget -O /app/yolov7_fire_af "https://www.dropbox.com/scl/fi/sy6hhmwycyfnxt3nmjilu/best_part_af?rlkey=466ulc6w8f7et4skp54v8inpu&st=v7dk7d81&dl=1"
RUN wget -O /app/yolov7_fire_ag "https://www.dropbox.com/scl/fi/2eq21o7moqwamlyr9oe47/best_part_ag?rlkey=grtbmqpa3bi7ez2ojyh4iwml6&st=3t6cykat&dl=1"
RUN wget -O /app/yolov7_fire_ah "https://www.dropbox.com/scl/fi/n7oqzb2qr06aqqrgoxpyy/best_part_ah?rlkey=6xvljm6xvb7fh77q35qfyl8wv&st=ecc26gt3&dl=1"
RUN wget -O /app/yolov7_fire_ai "https://www.dropbox.com/scl/fi/vx91vmlohxy48barilbhb/best_part_ai?rlkey=4ifs2h7809r6bramrnokflomw&st=lkuhk45f&dl=1"
RUN wget -O /app/yolov7_fire_aj "https://www.dropbox.com/scl/fi/sl85ajfj61fq7lpd28tsf/best_part_aj?rlkey=sz6rv7toee11op4con7ufjufn&st=nwwtb2lw&dl=1"
RUN wget -O /app/yolov7_fire_ak "https://www.dropbox.com/scl/fi/16zkz1hj0roi4996fk44c/best_part_ak?rlkey=djt5ev531sgc0a6z9rla5115t&st=jt8umqnl&dl=1"
RUN wget -O /app/yolov7_fire_al "https://www.dropbox.com/scl/fi/2v1iquko0q06xeoulkb2e/best_part_al?rlkey=mpn5nhe43xk25hl067htmvxks&st=p9x3sqvv&dl=1"
RUN wget -O /app/yolov7_fire_am "https://www.dropbox.com/scl/fi/5vkamze3fgjuq4t8eq9f4/best_part_am?rlkey=rs2bf7hqjqoytncq5r2a2tr4p&st=1u81boo4&dl=1"
RUN wget -O /app/yolov7_fire_an "https://www.dropbox.com/scl/fi/q2sxpcsdevpalti53f6b5/best_part_an?rlkey=jca0mvau6llv8e4oejz5t6heo&st=4jxvvrei&dl=1"
RUN curl -L -o /app/yolov7_fire_ao "https://www.dropbox.com/scl/fi/ep6m2jnhbp2wicvq1jy2s/best_part_ao?rlkey=apxxqht912box0sne42qa9ze9&st=adb8pct3&dl=1"

RUN cat /app/yolov7_fire_* > /app/yolov7-fire.pt

# RUN curl -L -o /app/model.pt "https://web.lcrc.anl.gov/public/waggle/models/vehicletracking/yolov7.pt"

ENTRYPOINT ["python3", "-u", "/app/app.py"]