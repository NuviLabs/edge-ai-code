from onnx_tf.backend import prepare
import onnx
import tensorflow_addons as tfa  # Ensure TensorFlow Addons is imported

# Load ONNX model
model_name = "best_model_yolo_m_tflite4"
# model_name = "model"
onnx_model = onnx.load(f"/home/kaeun.kim/kaeun-dev/nuvilab/models/export/{model_name}.onnx")

# Convert ONNX to TensorFlow SavedModel
tf_rep = prepare(onnx_model)
tf_rep.export_graph(f"{model_name}.tf")
