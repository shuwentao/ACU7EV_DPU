import tensorflow as tf
from tensorflow_model_optimization.quantization.keras import vitis_inspect

# ---- 绕过 vitis_inspect 已知 bug ----
# 工具在给模型插入输入量化层(quant_xxx)后，该层的“元数据键”与模型重建后的
# 实际层名对不上，原代码用 logger.error(...) 直接 raise，导致 inspect 中断。
# 这里给缺失元数据的层补一个默认 InspectResult 即可正常出报告；该层不对应
# 任何浮点层，跳过不影响检查结论。
_orig_extract = vitis_inspect.VitisInspector._extract_inspect_results

def _patched_extract(self, float_model, inspect_model, layer_metadata):
    for layer in inspect_model.layers:
        if layer.name not in layer_metadata:
            layer_metadata[layer.name] = {
                'inspect_result': vitis_inspect.InspectResult(
                    device='DPU', origin_layers=[])
            }
    return _orig_extract(self, float_model, inspect_model, layer_metadata)

vitis_inspect.VitisInspector._extract_inspect_results = _patched_extract

model = tf.keras.models.load_model('mnist_model.h5')

inspector = vitis_inspect.VitisInspector(target="DPUCZDX8G_ISA1_B4096")   # Zynq MPSoC 的 DPU
#inspector = vitis_inspect.VitisInspector(target="DPUCADF8H_ISA0")   # Alveo 卡（非 Zynq MPSoC）
inspector.inspect_model(model,
                        plot=True,
                        plot_file="model.svg",
                        dump_results=True,
                        dump_results_file="inspect_results.txt",
                        verbose=0)
