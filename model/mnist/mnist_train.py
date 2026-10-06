#!/usr/bin/env python3
"""MNIST 手写数字识别训练脚本（TensorFlow 2.x / Keras）。

适用：在 Anaconda 环境中直接运行
    conda activate <你的环境>
    python mnist_train.py

脚本会：
  1. 自动加载 MNIST 数据集（首次运行会下载到 ~/.keras/datasets）
  2. 构建一个轻量 CNN
  3. 训练并在测试集上评估
  4. 保存训练好的模型到 ./mnist_model.h5
"""
import ctypes
import glob
import os
import site

# 必须在 import tensorflow 之前设置，屏蔽冗余日志
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")


def preload_cuda_libs() -> None:
    """提前 dlopen pip 安装的 nvidia-* CUDA 库（与 tf_env_test.py 同款处理），
    避免 "Cannot dlopen some GPU libraries" 导致无法发现 GPU。"""
    candidates = []
    try:
        candidates.append(site.getusersitepackages())
    except Exception:
        pass
    candidates.extend(site.getsitepackages())
    for base in candidates:
        for lib in glob.glob(os.path.join(base, "nvidia", "*", "lib", "*.so*")):
            try:
                ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                pass


preload_cuda_libs()

import tensorflow as tf
from tensorflow.keras import layers, models

# 强制只用 CPU 训练：隐藏所有 GPU 设备
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    tf.config.set_visible_devices([], "GPU")
    print(f"已检测到 {len(gpus)} 块 GPU，但按要求强制使用 CPU 训练。")

EPOCHS = 5
BATCH_SIZE = 128
MODEL_PATH = "mnist_model.h5"


def load_data():
    """加载并预处理 MNIST。"""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # 归一化到 [0, 1]，并扩展通道维度 (N, 28, 28, 1)
    x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
    x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

    # 标签转 one-hot
    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)
    return (x_train, y_train), (x_test, y_test)


def build_model():
    """构建一个轻量 CNN。"""
    m = models.Sequential(
        [
            #layers.Input(shape=(28, 28, 1)),            # 入口：输入 28×28 单通道灰度图
            layers.Conv2D(32, 3, activation="relu",input_shape=(28,28,1)),    # 卷积层：32 个 3×3 滤波器，提取边缘等低层特征 → 26×26×32
            layers.MaxPooling2D(2),                    # 最大池化：2×2 降采样，尺寸减半 → 13×13×32
            layers.Conv2D(64, 3, activation="relu"),    # 卷积层：64 个 3×3 滤波器，提取更抽象高层特征 → 11×11×64
            layers.MaxPooling2D(2),                    # 再次池化，尺寸减半 → 5×5×64
            layers.Flatten(),                          # 展平：5×5×64 = 1600 个一维特征，供全连接层使用
            layers.Dense(128, activation="relu"),      # 全连接层：128 神经元，做高级特征组合
            layers.Dropout(0.5),                       # 随机丢弃 50% 神经元，防止过拟合
            layers.Dense(10, activation="softmax"),    # 输出层：10 类（0-9），softmax 输出概率分布
        ]
    )
    m.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return m


def main() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    print(f"TensorFlow 版本 : {tf.__version__}")
    print(f"训练设备       : {'GPU' if gpus else 'CPU'}\n")

    (x_train, y_train), (x_test, y_test) = load_data()
    print(f"训练样本: {x_train.shape[0]}, 测试样本: {x_test.shape[0]}\n")

    model = build_model()
    model.summary()

    print("\n开始训练 ...")
    model.fit(
        x_train,              # 训练输入：图片特征，形状 (N, 28, 28, 1)
        y_train,              # 训练标签：每张图对应的正确答案（one-hot）
        batch_size=BATCH_SIZE,  # 批次大小：每次用 128 张图算一次梯度更新
        epochs=EPOCHS,         # 训练轮数：把全部训练数据完整过 5 遍
        validation_split=0.1,  # 自动划出 10% 训练数据当验证集（不参与训练，只评估泛化）
        verbose=1,             # 日志模式：1=带进度条的实时日志；0=静默；2=每轮一行汇总
    )

    print("\n在测试集上评估 ...")
    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"测试集损失: {loss:.4f}, 测试集准确率: {acc * 100:.2f}%")

    model.save(MODEL_PATH)
    print(f"\n模型已保存到 {MODEL_PATH}")


if __name__ == "__main__":
    main()
