import cv2
import os
import tensorflow as tf
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Đường dẫn đến thư mục ảnh
data_dir = '/home/kasumi/Home/code/group_project/dataset/ki_yeu'
not_face_dir = '/home/kasumi/Home/code/group_project/dataset/ki_yeu/not_face'

# Tạo danh sách các đường dẫn đến ảnh và nhãn tương ứng
face_images = [os.path.join(data_dir, img) for img in os.listdir(data_dir) if img.endswith('.jpg')]
face_labels = ['face'] * len(face_images)

not_face_images = [os.path.join(not_face_dir, img) for img in os.listdir(not_face_dir) if img.endswith('.jpg')]
not_face_labels = ['not_face'] * len(not_face_images)

# Đảm bảo có ít nhất một ảnh thuộc mỗi lớp
if not not_face_images:
    raise ValueError("Không tìm thấy ảnh nào không phải khuôn mặt. Vui lòng thêm ảnh vào thư mục này.")

# Gộp danh sách ảnh và nhãn
image_paths = face_images + not_face_images
labels = face_labels + not_face_labels

# Chia dữ liệu thành tập huấn luyện và kiểm tra
train_paths, val_paths, train_labels, val_labels = train_test_split(image_paths, labels, test_size=0.2, stratify=labels)

# Tạo DataFrame từ danh sách
train_df = pd.DataFrame({'filename': train_paths, 'class': train_labels})
val_df = pd.DataFrame({'filename': val_paths, 'class': val_labels})

# Tạo trình tạo dữ liệu hình ảnh
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=None,
    x_col='filename',
    y_col='class',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_df,
    directory=None,
    x_col='filename',
    y_col='class',
    target_size=(150, 150),
    batch_size=32,
    class_mode='binary',
    shuffle=True
)

# Tạo mô hình
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Biên dịch mô hình
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Hàm callback để hiển thị số lượng ảnh đã train
class DisplayImagesTrained(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        num_images = train_generator.samples  # Số lượng ảnh thực tế trong tập huấn luyện
        print(f"Số lượng ảnh đã train ở epoch {epoch+1}: {num_images}")

# Đào tạo mô hình với callback
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator,
    callbacks=[DisplayImagesTrained()]
)

# Đánh giá mô hình và tính toán % khuôn mặt được nhận diện đúng
val_loss, val_accuracy = model.evaluate(val_generator)
print(f"Accuracy: {val_accuracy * 100:.2f}%")

