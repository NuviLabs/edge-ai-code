import onnx

# Load the ONNX model
onnx_model = onnx.load("/home/kaeun.kim/kaeun-dev/nuvilab/models/export/resnet18_binary_static.onnx")

# Print model input/output shapes
for input in onnx_model.graph.input:
    print(f"Input: {input.name}, Shape: {[dim.dim_value for dim in input.type.tensor_type.shape.dim]}")

for output in onnx_model.graph.output:
    print(f"Output: {output.name}, Shape: {[dim.dim_value for dim in output.type.tensor_type.shape.dim]}")
