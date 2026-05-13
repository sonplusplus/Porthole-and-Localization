import onnxruntime as ort

sess = ort.InferenceSession(
    r"models\depth_anything_v2_vits.onnx"
)

print(sess.get_inputs()[0].shape)
print(sess.get_outputs()[0].shape)