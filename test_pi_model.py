import os
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter

model_path = os.path.expanduser(
    "~/mnist_edge_ai/Laptop/models/mnist_cnn_int8.tflite"
)

interpreter = Interpreter(model_path=model_path, num_threads=1)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

print("Model loaded successfully")
print("Model size:", os.path.getsize(model_path), "bytes")
print("Input:", inp["shape"], inp["dtype"])
print("Output:", out["shape"], out["dtype"])

# Blank-image smoke test; this verifies inference runs,
# not whether the predicted digit is correct.
image = np.full((1, 28, 28, 1), -128, dtype=np.int8)
interpreter.set_tensor(inp["index"], image)

start = time.perf_counter()
interpreter.invoke()
elapsed_ms = (time.perf_counter() - start) * 1000

scores = interpreter.get_tensor(out["index"])[0]
scale, zero = out["quantization"]
scores = (scores.astype(np.float32) - zero) * scale

print("Inference successful")
print("Predicted digit for blank image:", int(np.argmax(scores)))
print("Inference latency: %.3f ms" % elapsed_ms)
print("Output scores:", scores)
