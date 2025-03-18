import subprocess
subprocess.Popen("/home/phatnguyen/Documents/repo/self-driving_car_using_CNN/term1-simulator-linux/beta_simulator_linux/beta_simulator.x86_64")

# Thêm các thư viện cần thiết
from data_preprocessing import *
import socketio
import eventlet
import numpy as np
from flask import Flask
from tensorflow.keras.models import load_model
import base64
from io import BytesIO
from PIL import Image
import cv2
import os
# Tắt thông báo các lỗi
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

MAX_SPEED = 15
MIN_SPEED = 12
speed_limit = MAX_SPEED
sio = socketio.Server()

app = Flask(__name__) # '__main__'

@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # Lấy giá trị throttle hiện tại
        throttle = float(data["throttle"])
        # Góc lái hiện tại của ô tô
        steering_angle = float(data["steering_angle"])
    	  # Tốc độ hiện tại của ô tô
        speed = float(data["speed"])
        # Ảnh từ camera giữa
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
			# Tiền xử lý ảnh, cắt, reshape
            image = np.asarray(image)       
            image = image_preprocessing(image)
            image = np.array([image])
            print('*****************************************************')
            steering_angle = float(model.predict(image, batch_size=1))
            
			# Tốc độ ta để trong khoảng từ 10 đến 25
            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED  # giảm tốc độ
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - (speed/speed_limit)**2 - 0.15

            print('{} {} {}'.format(steering_angle, throttle, speed))
			
			# Gửi lại dữ liệu về góc lái, tốc độ cho phần mềm để ô tô tự lái
            sendControl(steering_angle, throttle)
        except Exception as e:
            print(e)
    else:
        sio.emit('manual', data={}, skip_sid=True)
    
@sio.on('connect')
def connect(sid, environ):
    print('Connected')
    sendControl(0, 0)


def sendControl(steering, throttle):
    sio.emit('steer', data = {
        'steering_angle': steering.__str__(),
        'throttle': throttle.__str__()
    })


if __name__ == '__main__':
    model = load_model('/home/phatnguyen/Documents/repo/self-driving_car_using_CNN/model.h5')
    app = socketio.Middleware(sio, app)
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
