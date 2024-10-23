from retinaface import RetinaFace
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import custom_object_scope
import numpy as np
import cv2
import os

# Hàm load dữ liệu với việc scale bounding boxes đúng cách
def load_data(data_dir, target_size=(128, 128)):
    images = []
    boxes = []
    
    for file in os.listdir(data_dir):
        if file.endswith('.txt'):  # chỉ lấy các file _ldmks.txt
            with open(os.path.join(data_dir, file), 'r') as f:
                # Assuming box format: x_min, y_min, x_max, y_max (absolute coordinates)
                box = list(map(float, f.read().strip().split()))
                
                img_file = file.replace('.txt', '.jpg')
                img_path = os.path.join(data_dir, img_file)
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    orig_height, orig_width = img.shape[:2]
                    
                    # Resize image
                    img_resized = cv2.resize(img, target_size)
                    images.append(img_resized)
                    
                    # Calculate scaling factors
                    scale_x = target_size[1] / orig_width
                    scale_y = target_size[0] / orig_height
                    
                    # Scale bounding box accordingly
                    box_resized = [
                        box[0] * scale_x,  # x_min
                        box[1] * scale_y,  # y_min
                        box[2] * scale_x,  # x_max
                        box[3] * scale_y   # y_max
                    ]
                    boxes.append(box_resized)
                    
    return np.array(images), np.array(boxes)

# Load the dataset (with resized images and scaled bounding boxes)
data_dir = '/home/kasumi/Home/code/group_project/Main/dataset/temp'
image_height, image_width = 128, 128
images, boxes = load_data(data_dir, target_size=(image_height, image_width))

# Define the custom RetinaFace Layer (mock implementation)
class RetinaFaceLayer(Layer):
    def __init__(self, **kwargs):
        super(RetinaFaceLayer, self).__init__(**kwargs)
        self.model = RetinaFace.build_model()  # Pre-trained RetinaFace model
    
    def call(self, inputs):
        outputs = self.model(inputs)
        bbox_outputs = outputs[0]  # Assuming bbox is the first output of the model
        return bbox_outputs

# Build the model
input_layer = Input(shape=(image_height, image_width, 3))
retina_face_layer = RetinaFaceLayer()(input_layer)
global_avg_pooling_layer = GlobalAveragePooling2D()(retina_face_layer)
output_layer = Dense(4, activation='linear')(global_avg_pooling_layer)  # 4 values for bbox (x_min, y_min, x_max, y_max)

fine_tuned_model = Model(inputs=input_layer, outputs=output_layer)
fine_tuned_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
fine_tuned_model.fit(images, boxes, epochs=10, batch_size=8)

# Save the model with the custom RetinaFaceLayer
with custom_object_scope({'RetinaFaceLayer': RetinaFaceLayer}):
    fine_tuned_model.save('/home/kasumi/Home/code/group_project/Main/retinaface/model.h5')
