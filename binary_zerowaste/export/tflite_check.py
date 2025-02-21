import numpy as np
import tensorflow as tf

# Load the TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test with dummy input
input_shape = input_details[0]['shape']
dummy_input = np.random.rand(*input_shape).astype(np.float32)

interpreter.set_tensor(input_details[0]['index'], dummy_input)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

print("Output:", output_data)
