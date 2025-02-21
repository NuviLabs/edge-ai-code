import tensorflow as tf

saved_model_dir = "best_model_yolo_m_tflite4.tf"
tflite_model_path = "best_model_yolo_m_tflite4.tflite"

# Convert the SavedModel to TFLite format
converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

# Save the TFLite model
with open(tflite_model_path, "wb") as f:
    f.write(tflite_model)

print("TFLite model saved at:", tflite_model_path)
