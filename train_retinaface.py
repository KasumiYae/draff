from retinaface import RetinaFace
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Layer, Flatten, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import custom_object_scope
import numpy as np
import cv2
import os

# Hàm load dữ liệu
def load_data(data_dir):
    images = []
    boxes = []
    for file in os.listdir(data_dir):
        if file.endswith('_ldmks.txt'):  # chỉ lấy các file _ldmks.txt
            with open(os.path.join(data_dir, file), 'r') as f:
                box = list(map(float, f.read().strip().split()))
                img_file = file.replace('_ldmks.txt', '.png')
                img_path = os.path.join(data_dir, img_file)
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    images.append(img)
                    boxes.append(box)
    return np.array(images), np.array(boxes)

data_dir = '/home/kasumi/Home/code/group_project/Main/dataset/dataset_100'
images, boxes = load_data(data_dir)

# Tạo mô hình RetinaFace đã tinh chỉnh
class RetinaFaceLayer(Layer):
    def __init__(self, **kwargs):
        super(RetinaFaceLayer, self).__init__(**kwargs)
        self.model = RetinaFace.build_model()

    def call(self, inputs):
        outputs = self.model(inputs)
        bbox_outputs = outputs[0]  # Giả định đầu ra đầu tiên là bbox
        return bbox_outputs

input_layer = Input(shape=(None, None, 3))
retina_face_layer = RetinaFaceLayer()(input_layer)
global_avg_pooling_layer = GlobalAveragePooling2D()(retina_face_layer)
output_layer = Dense(140, activation='linear')(global_avg_pooling_layer)  # 140 giá trị bounding box

fine_tuned_model = Model(inputs=input_layer, outputs=output_layer)
fine_tuned_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Resize các hình ảnh cho phù hợp với input của mô hình
image_height, image_width = 128, 128
images_resized = np.array([cv2.resize(img, (image_width, image_height)) for img in images])

# Huấn luyện mô hình
fine_tuned_model.fit(images_resized, boxes, epochs=10, batch_size=8)

# Lưu mô hình trong custom_object_scope
with custom_object_scope({'RetinaFaceLayer': RetinaFaceLayer}):
    fine_tuned_model.save('/home/kasumi/Home/code/group_project/Main/retinaface/fine_tuned_retinaface.h5')
