import tensorflow as tf
import numpy as np
from tensorflow_model_optimization.quantization.keras import vitis_quantize

# 1) 加载干净浮点模型
model = tf.keras.models.load_model('mnist_model.h5')

# 2) 准备校准集（与训练一致的预处理：/255，shape (N,28,28,1)）
(_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_calib = x_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

calib_ds = tf.data.Dataset.from_tensor_slices(x_calib[:100]).batch(1)  # 100 张校准

# 3) 量化（不走 inspect，避免那个元数据 bug）
quantizer = vitis_quantize.VitisQuantizer(model, target='DPUCZDX8G_ISA1_B4096')
quantized_model = quantizer.quantize_model(
    calib_dataset=calib_ds,
    calib_steps=100,
    verbose=1)

# 4) 保存量化模型
quantized_model.save('mnist_quantized.h5')
print('quantized model saved -> mnist_quantized.h5')

