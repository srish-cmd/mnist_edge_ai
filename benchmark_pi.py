import os
import sys
import time
import numpy as np
from tflite_runtime.interpreter import Interpreter

if len(sys.argv) != 2:
    print("Usage: python benchmark_pi.py MODEL_PATH")
    sys.exit(1)

model_path = sys.argv[1]

def get_rss_kb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1])
    return 0

rss_before = get_rss_kb()
model_size = os.path.getsize(model_path)

interpreter = Interpreter(model_path=model_path, num_threads=1)
interpreter.allocate_tensors()

inp = interpreter.get_input_details()[0]
out = interpreter.get_output_details()[0]
rss_after = get_rss_kb()

image = np.zeros(inp["shape"], dtype=inp["dtype"])

# Warm up the interpreter
for _ in range(20):
    interpreter.set_tensor(inp["index"], image)
    interpreter.invoke()

# Measure 500 inference runs
times = []
for _ in range(500):
    interpreter.set_tensor(inp["index"], image)
    start = time.perf_counter()
    interpreter.invoke()
    times.append((time.perf_counter() - start) * 1000)

print("\n--- RASPBERRY PI BENCHMARK ---")
print("Model:", os.path.basename(model_path))
print("Model size:", model_size, "bytes")
print("Input dtype:", inp["dtype"])
print("Output dtype:", out["dtype"])
print("RSS before allocation:", round(rss_before / 1024, 2), "MiB")
print("RSS after allocation:", round(rss_after / 1024, 2), "MiB")
print("Approx. RSS increase:", round((rss_after - rss_before) / 1024, 2), "MiB")
print("Warm-up runs: 20")
print("Timed runs: 500")
print("Mean latency:", round(float(np.mean(times)), 3), "ms")
print("Median latency:", round(float(np.median(times)), 3), "ms")
print("P95 latency:", round(float(np.percentile(times, 95)), 3), "ms")
