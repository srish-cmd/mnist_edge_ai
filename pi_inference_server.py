import json
import os
import socket
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter

MODEL = os.path.expanduser(
    "~/mnist_edge_ai/Laptop/models/mnist_cnn_int8.tflite"
)
HOST = "0.0.0.0"
PORT = 5000

interpreter = Interpreter(model_path=MODEL, num_threads=1)
interpreter.allocate_tensors()
inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]

print("INT8 model loaded:", os.path.getsize(MODEL), "bytes", flush=True)
print("Listening on port", PORT, flush=True)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server:
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)

    while True:
        conn, addr = server.accept()
        with conn:
            try:
                data = bytearray()
                while len(data) < 784:
                    chunk = conn.recv(784 - len(data))
                    if not chunk:
                        break
                    data.extend(chunk)

                if len(data) != 784:
                    conn.sendall(b'{"error":"expected 784 image bytes"}\n')
                    continue

                image = np.frombuffer(bytes(data), dtype=np.int8)
                image = image.reshape(1, 28, 28, 1)

                interpreter.set_tensor(inp["index"], image)
                start = time.perf_counter()
                interpreter.invoke()
                latency_ms = (time.perf_counter() - start) * 1000

                quantized = interpreter.get_tensor(out["index"])[0]
                scale, zero = out["quantization"]
                scores = (quantized.astype(np.float32) - zero) * scale
                digit = int(np.argmax(scores))

                response = {
                    "digit": digit,
                    "confidence": float(scores[digit]),
                    "latency_ms": round(latency_ms, 3)
                }
                conn.sendall((json.dumps(response) + "\n").encode())
                print(addr, response, flush=True)

            except Exception as exc:
                print("Request error:", exc, flush=True)
                try:
                    conn.sendall(b'{"error":"inference failed"}\n')
                except OSError:
                    pass
