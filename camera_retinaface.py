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
model = load_model('/home/kasumi/Home/code/group_project/Main/retinaface/model.h5', custom_objects={'RetinaFaceLayer': RetinaFaceLayer})

# Kích thước input của mô hình (cần phải giống với kích thước mà mô hình được huấn luyện)
input_size = (128, 128)

# Hàm để nhận diện khuôn mặt từ một khung hình sử dụng mô hình đã fine-tuned
def detect_face(frame):
    # Resize frame to match model input size
    resized_frame = cv2.resize(frame, input_size)
    
    # Normalize the frame (if necessary, e.g., scaling pixel values between 0-1)
    normalized_frame = resized_frame / 255.0
    
    # Add batch dimension (required by the model)
    input_tensor = np.expand_dims(normalized_frame, axis=0)
    
    # Predict bounding boxes using the fine-tuned model
    predicted_boxes = model.predict(input_tensor)[0]  # [0] to remove batch dimension
    
    # Scale predicted boxes back to the original frame size
    height, width, _ = frame.shape
    scale_x = width / input_size[0]
    scale_y = height / input_size[1]
    
    # Scale the bounding box coordinates
    x_min = int(predicted_boxes[0] * scale_x)
    y_min = int(predicted_boxes[1] * scale_y)
    x_max = int(predicted_boxes[2] * scale_x)
    y_max = int(predicted_boxes[3] * scale_y)
    
    return (x_min, y_min, x_max, y_max)

# Khởi tạo camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Nhận diện khuôn mặt từ khung hình
    x_min, y_min, x_max, y_max = detect_face(frame)
    
    # Vẽ bounding box quanh khuôn mặt
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    
    # Hiển thị khung hình
    cv2.imshow('Face Detection', frame)
    
    # Nhấn 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()
