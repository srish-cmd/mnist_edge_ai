import os
import numpy as np
import tensorflow as tf


# --------------------------------------------------
# Configuration
# --------------------------------------------------

KERAS_MODEL_PATH = "../models/mnist_cnn_fp32.keras"

FP32_MODEL_PATH = "../models/mnist_cnn_fp32.tflite"
INT8_MODEL_PATH = "../models/mnist_cnn_int8.tflite"

REPRESENTATIVE_SAMPLES = 500


# --------------------------------------------------
# 1. Load trained Keras model
# --------------------------------------------------

print("Loading trained model...")

model = tf.keras.models.load_model(KERAS_MODEL_PATH)

print("Model loaded successfully.")


# --------------------------------------------------
# 2. Load MNIST data for representative dataset
# --------------------------------------------------

print("\nLoading MNIST dataset...")

(x_train, _), (_, _) = tf.keras.datasets.mnist.load_data()

# Convert to float32 [0, 1]
x_train = x_train.astype(np.float32) / 255.0

# Add channel dimension
x_train = np.expand_dims(x_train, axis=-1)

print(f"MNIST training data shape: {x_train.shape}")


# --------------------------------------------------
# 3. Convert to FP32 LiteRT/TFLite model
# --------------------------------------------------

print("\nConverting FP32 model...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

fp32_tflite_model = converter.convert()

with open(FP32_MODEL_PATH, "wb") as f:
    f.write(fp32_tflite_model)

print(f"FP32 model saved to: {FP32_MODEL_PATH}")


# --------------------------------------------------
# 4. Representative dataset generator
# --------------------------------------------------

def representative_dataset():
    for image in x_train[:REPRESENTATIVE_SAMPLES]:
        image = np.expand_dims(image, axis=0)
        yield [image]


# --------------------------------------------------
# 5. Convert to fully INT8 model
# --------------------------------------------------

print("\nConverting INT8 quantized model...")

converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Enable default post-training quantization
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Representative dataset for calibration
converter.representative_dataset = representative_dataset

# Force all supported operations to INT8
converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

# INT8 input and output
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

int8_tflite_model = converter.convert()

with open(INT8_MODEL_PATH, "wb") as f:
    f.write(int8_tflite_model)

print(f"INT8 model saved to: {INT8_MODEL_PATH}")


# --------------------------------------------------
# 6. Compare model sizes
# --------------------------------------------------

fp32_size = os.path.getsize(FP32_MODEL_PATH)
int8_size = os.path.getsize(INT8_MODEL_PATH)

print("\n----------------------------------------")
print("MODEL SIZE COMPARISON")
print("----------------------------------------")

print(f"FP32 model: {fp32_size:,} bytes ({fp32_size / 1024:.2f} KB)")
print(f"INT8 model: {int8_size:,} bytes ({int8_size / 1024:.2f} KB)")

reduction = (1 - int8_size / fp32_size) * 100

print(f"Size reduction: {reduction:.2f}%")
print("----------------------------------------")