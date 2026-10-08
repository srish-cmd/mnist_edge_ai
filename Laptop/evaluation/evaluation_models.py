import os
import time
import gc
import numpy as np
import tensorflow as tf
import psutil


# ============================================================
# Configuration
# ============================================================

FP32_MODEL_PATH = "../models/mnist_cnn_fp32.tflite"
INT8_MODEL_PATH = "../models/mnist_cnn_int8.tflite"

WARMUP_RUNS = 20
BENCHMARK_SAMPLES = 1000


# ============================================================
# Load MNIST test data
# ============================================================

print("Loading MNIST test dataset...")

(_, _), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_test = x_test.astype(np.float32) / 255.0
x_test = np.expand_dims(x_test, axis=-1)

print(f"Test images: {x_test.shape}")
print(f"Test labels: {y_test.shape}")


# ============================================================
# Helper: Load LiteRT/TFLite model
# ============================================================

def load_interpreter(model_path):

    interpreter = tf.lite.Interpreter(
        model_path=model_path,
        num_threads=1
    )

    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    return interpreter, input_details, output_details


# ============================================================
# Helper: Quantize input for INT8 model
# ============================================================

def quantize_input(image, input_details):

    scale, zero_point = input_details[0]["quantization"]

    quantized = np.round(image / scale + zero_point)

    quantized = np.clip(
        quantized,
        -128,
        127
    ).astype(np.int8)

    return quantized


# ============================================================
# Helper: Dequantize INT8 output
# ============================================================

def dequantize_output(output, output_details):

    scale, zero_point = output_details[0]["quantization"]

    return (output.astype(np.float32) - zero_point) * scale


# ============================================================
# Evaluate accuracy
# ============================================================

def evaluate_accuracy(
    interpreter,
    input_details,
    output_details,
    model_type
):

    correct = 0

    start_time = time.perf_counter()

    for i in range(len(x_test)):

        image = x_test[i:i + 1]

        # Prepare input
        if model_type == "INT8":
            input_data = quantize_input(
                image,
                input_details
            )
        else:
            input_data = image.astype(np.float32)

        interpreter.set_tensor(
            input_details[0]["index"],
            input_data
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details[0]["index"]
        )

        # Convert INT8 output back to float
        if model_type == "INT8":
            output = dequantize_output(
                output,
                output_details
            )

        prediction = np.argmax(output, axis=1)[0]

        if prediction == y_test[i]:
            correct += 1

    total_time = time.perf_counter() - start_time

    accuracy = correct / len(x_test)

    return accuracy, total_time


# ============================================================
# Measure inference latency
# ============================================================

def benchmark_latency(
    interpreter,
    input_details,
    output_details,
    model_type
):

    latencies = []

    # --------------------------------------------------------
    # Warm-up
    # --------------------------------------------------------

    print(f"Running {WARMUP_RUNS} warm-up inferences...")

    for i in range(WARMUP_RUNS):

        image = x_test[i:i + 1]

        if model_type == "INT8":
            input_data = quantize_input(
                image,
                input_details
            )
        else:
            input_data = image.astype(np.float32)

        interpreter.set_tensor(
            input_details[0]["index"],
            input_data
        )

        interpreter.invoke()

        interpreter.get_tensor(
            output_details[0]["index"]
        )

    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    print(
        f"Benchmarking {BENCHMARK_SAMPLES} "
        f"inferences..."
    )

    for i in range(BENCHMARK_SAMPLES):

        image = x_test[i:i + 1]

        if model_type == "INT8":
            input_data = quantize_input(
                image,
                input_details
            )
        else:
            input_data = image.astype(np.float32)

        start = time.perf_counter()

        interpreter.set_tensor(
            input_details[0]["index"],
            input_data
        )

        interpreter.invoke()

        interpreter.get_tensor(
            output_details[0]["index"]
        )

        end = time.perf_counter()

        latency_ms = (end - start) * 1000

        latencies.append(latency_ms)

    latencies = np.array(latencies)

    return {
        "average": np.mean(latencies),
        "median": np.median(latencies),
        "p95": np.percentile(latencies, 95),
        "minimum": np.min(latencies),
        "maximum": np.max(latencies)
    }


# ============================================================
# Memory measurement
# ============================================================

def get_memory_mb():

    process = psutil.Process(
        os.getpid()
    )

    memory_bytes = process.memory_info().rss

    return memory_bytes / (1024 * 1024)


# ============================================================
# Model evaluation
# ============================================================

def evaluate_model(model_path, model_type):

    print("\n")
    print("=" * 60)
    print(f"Evaluating {model_type} model")
    print("=" * 60)

    # --------------------------------------------------------
    # Model size
    # --------------------------------------------------------

    model_size_bytes = os.path.getsize(model_path)

    model_size_kb = model_size_bytes / 1024

    print(
        f"Model size: "
        f"{model_size_bytes:,} bytes "
        f"({model_size_kb:.2f} KB)"
    )

    # --------------------------------------------------------
    # Memory before loading
    # --------------------------------------------------------

    memory_before = get_memory_mb()

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    interpreter, input_details, output_details = (
        load_interpreter(model_path)
    )

    memory_after_load = get_memory_mb()

    print(
        f"Memory before model load: "
        f"{memory_before:.2f} MB"
    )

    print(
        f"Memory after model load: "
        f"{memory_after_load:.2f} MB"
    )

    model_memory = memory_after_load - memory_before

    print(
        f"Approx. model memory increase: "
        f"{model_memory:.2f} MB"
    )

    # --------------------------------------------------------
    # Display input/output information
    # --------------------------------------------------------

    print("\nInput details:")
    print(f"Shape: {input_details[0]['shape']}")
    print(f"Data type: {input_details[0]['dtype']}")
    print(
        f"Quantization: "
        f"{input_details[0]['quantization']}"
    )

    print("\nOutput details:")
    print(f"Shape: {output_details[0]['shape']}")
    print(f"Data type: {output_details[0]['dtype']}")
    print(
        f"Quantization: "
        f"{output_details[0]['quantization']}"
    )

    # --------------------------------------------------------
    # Accuracy
    # --------------------------------------------------------

    print("\nEvaluating accuracy...")

    accuracy, evaluation_time = evaluate_accuracy(
        interpreter,
        input_details,
        output_details,
        model_type
    )

    print(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Evaluation time: "
        f"{evaluation_time:.2f} seconds"
    )

    # --------------------------------------------------------
    # Latency
    # --------------------------------------------------------

    latency = benchmark_latency(
        interpreter,
        input_details,
        output_details,
        model_type
    )

    print("\nLatency results:")
    print(
        f"Average:  {latency['average']:.4f} ms"
    )
    print(
        f"Median:   {latency['median']:.4f} ms"
    )
    print(
        f"P95:      {latency['p95']:.4f} ms"
    )
    print(
        f"Minimum:  {latency['minimum']:.4f} ms"
    )
    print(
        f"Maximum:  {latency['maximum']:.4f} ms"
    )

    # --------------------------------------------------------
    # Final memory
    # --------------------------------------------------------

    memory_after_benchmark = get_memory_mb()

    print(
        f"\nMemory after benchmark: "
        f"{memory_after_benchmark:.2f} MB"
    )

    # Clean up
    del interpreter
    gc.collect()

    return {
        "accuracy": accuracy * 100,
        "size_bytes": model_size_bytes,
        "size_kb": model_size_kb,
        "memory_increase_mb": model_memory,
        "average_latency_ms": latency["average"],
        "median_latency_ms": latency["median"],
        "p95_latency_ms": latency["p95"]
    }


# ============================================================
# Main
# ============================================================

print("\n")
print("#" * 60)
print("MNIST FP32 vs INT8 LiteRT BENCHMARK")
print("#" * 60)

fp32_results = evaluate_model(
    FP32_MODEL_PATH,
    "FP32"
)

int8_results = evaluate_model(
    INT8_MODEL_PATH,
    "INT8"
)


# ============================================================
# Comparison
# ============================================================

size_reduction = (
    1 -
    int8_results["size_bytes"] /
    fp32_results["size_bytes"]
) * 100

accuracy_difference = (
    int8_results["accuracy"] -
    fp32_results["accuracy"]
)

latency_change = (
    1 -
    int8_results["average_latency_ms"] /
    fp32_results["average_latency_ms"]
) * 100


print("\n")
print("#" * 60)
print("FINAL FP32 vs INT8 COMPARISON")
print("#" * 60)

print(
    f"\n{'Metric':<30}"
    f"{'FP32':>15}"
    f"{'INT8':>15}"
)

print("-" * 60)

print(
    f"{'Accuracy':<30}"
    f"{fp32_results['accuracy']:>14.2f}%"
    f"{int8_results['accuracy']:>14.2f}%"
)

print(
    f"{'Model size (KB)':<30}"
    f"{fp32_results['size_kb']:>15.2f}"
    f"{int8_results['size_kb']:>15.2f}"
)

print(
    f"{'Average latency (ms)':<30}"
    f"{fp32_results['average_latency_ms']:>15.4f}"
    f"{int8_results['average_latency_ms']:>15.4f}"
)

print(
    f"{'Median latency (ms)':<30}"
    f"{fp32_results['median_latency_ms']:>15.4f}"
    f"{int8_results['median_latency_ms']:>15.4f}"
)

print(
    f"{'P95 latency (ms)':<30}"
    f"{fp32_results['p95_latency_ms']:>15.4f}"
    f"{int8_results['p95_latency_ms']:>15.4f}"
)

print(
    f"{'Memory increase (MB)':<30}"
    f"{fp32_results['memory_increase_mb']:>15.2f}"
    f"{int8_results['memory_increase_mb']:>15.2f}"
)

print("-" * 60)

print(
    f"\nModel size reduction: "
    f"{size_reduction:.2f}%"
)

print(
    f"Accuracy difference: "
    f"{accuracy_difference:+.2f} percentage points"
)

print(
    f"Average latency change: "
    f"{latency_change:+.2f}%"
)

print("\nBenchmark complete.")