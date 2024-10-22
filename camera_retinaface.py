from retinaface import RetinaFace
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Layer
import numpy as np
import cv2

# Định nghĩa lớp RetinaFaceLayer
class RetinaFaceLayer(Layer):
    def __init__(self, **kwargs):
        super(RetinaFaceLayer, self).__init__(**kwargs)
        self.model = RetinaFace.build_model()
    def call(self, inputs):
        outputs = self.model(inputs)
        bbox_outputs = outputs[0]  # Giả định đầu ra đầu tiên là bbox
        return bbox_outputs

# Tải mô hình đã fine-tuned
model = load_model('/home/kasumi/Home/code/group_project/Main/retinaface/fine_tuned_retinaface.h5', custom_objects={'RetinaFaceLayer': RetinaFaceLayer})

# Hàm để nhận diện khuôn mặt từ một khung hình
def detect_face(frame):
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    detections = RetinaFace.detect_faces(frame_rgb)
    return detections

# Khởi tạo camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Nhận diện khuôn mặt
    detections = detect_face(frame)
    
    # Vẽ bounding box quanh các khuôn mặt
    for key in detections:
        facial_area = detections[key]['facial_area']
        cv2.rectangle(frame, (facial_area[0], facial_area[1]), (facial_area[2], facial_area[3]), (0, 255, 0), 2)
    
    # Hiển thị khung hình
    cv2.imshow('Face Detection', frame)
    
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
