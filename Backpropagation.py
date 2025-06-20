# Cài đặt thư viện
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt

# Tiền xử lý dữ liệu MNIST
# Tải dữ liệu
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Chuẩn hóa giá trị ảnh về [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Thêm chiều kênh cho ảnh (batch, height, width, channels)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # One-hot encoding cho nhãn
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test

# Xây dựng mô hình mạng nơ-ron (CNN)
def build_model():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

# Huấn luyện mô hình
def train_model(model, x_train, y_train, use_early_stopping = False):
    callbacks = []
    if use_early_stopping:
        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=2,
            restore_best_weights=True
        )
        callbacks.append(early_stop)

    history = model.fit(
        x_train, y_train,
        epochs=20 if use_early_stopping else 5,
        batch_size=64,
        validation_split=0.1,
        verbose=1,
        callbacks=callbacks
    )
    return history

def plot_history(history, title_suffix=""):
    # Vẽ biểu đồ Loss / MSE
    plt.plot(history.history['loss'], label='Training')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss / Mean Squared Error')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Vẽ biểu đồ Accuracy
    plt.plot(history.history['accuracy'], label='Training')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy và Val_Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    model = build_model()
    
    print("-- Huấn luyện mô hình không dùng Early Stopping --")
    history = train_model(model, x_train, y_train, use_early_stopping=False)
    plot_history(history)

    print("-- Huấn luyện mô hình với Early Stopping --")
    model = build_model()  # rebuild to reset weights
    history = train_model(model, x_train, y_train, use_early_stopping=True)
    plot_history(history, title_suffix="With Early Stopping")

if __name__ == "__main__":
    main()