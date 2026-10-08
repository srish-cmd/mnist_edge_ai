import os
import time
import numpy as np
import tensorflow as tf


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

EPOCHS = 10
BATCH_SIZE = 128


# --------------------------------------------------
# 2. Load MNIST dataset
# --------------------------------------------------

print("Loading MNIST dataset...")

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

print(f"Training images: {x_train.shape}")
print(f"Training labels: {y_train.shape}")
print(f"Test images:     {x_test.shape}")
print(f"Test labels:     {y_test.shape}")


# --------------------------------------------------
# 3. Preprocess the images
# --------------------------------------------------

# Convert pixel values from [0, 255] to [0, 1]
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Add channel dimension:
# (60000, 28, 28) -> (60000, 28, 28, 1)
x_train = np.expand_dims(x_train, axis=-1)
x_test = np.expand_dims(x_test, axis=-1)


# --------------------------------------------------
# 4. Build lightweight CNN
# --------------------------------------------------

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28, 1)),

    tf.keras.layers.Conv2D(
        16,
        kernel_size=(3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Conv2D(
        32,
        kernel_size=(3, 3),
        activation="relu"
    ),

    tf.keras.layers.MaxPooling2D(
        pool_size=(2, 2)
    ),

    tf.keras.layers.Flatten(),

    tf.keras.layers.Dense(
        32,
        activation="relu"
    ),

    tf.keras.layers.Dense(
        10,
        activation="softmax"
    )
])


# --------------------------------------------------
# 5. Display model architecture
# --------------------------------------------------

print("\nModel architecture:")
model.summary()


# --------------------------------------------------
# 6. Compile model
# --------------------------------------------------

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)


# --------------------------------------------------
# 7. Train model
# --------------------------------------------------

print("\nStarting training...")

start_time = time.perf_counter()

history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_split=0.1,
    verbose=1
)

training_time = time.perf_counter() - start_time

print(f"\nTraining time: {training_time:.2f} seconds")


# --------------------------------------------------
# 8. Evaluate on MNIST test set
# --------------------------------------------------

print("\nEvaluating model on MNIST test set...")

test_loss, test_accuracy = model.evaluate(
    x_test,
    y_test,
    verbose=0
)

print(f"Test loss:     {test_loss:.4f}")
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


# --------------------------------------------------
# 9. Save trained model
# --------------------------------------------------

os.makedirs("../models", exist_ok=True)

model_path = "../models/mnist_cnn_fp32.keras"

model.save(model_path)

print(f"\nModel saved to: {model_path}")

# --------------------------------------------------
# 10. Save training results
# --------------------------------------------------

results_path = "../evaluation/training_results.txt"

with open(results_path, "w") as f:
    f.write("MNIST CNN Training Results\n")
    f.write("==========================\n")
    f.write(f"Epochs: {EPOCHS}\n")
    f.write(f"Batch size: {BATCH_SIZE}\n")
    f.write(f"Training time: {training_time:.2f} seconds\n")
    f.write(f"Test loss: {test_loss:.4f}\n")
    f.write(f"Test accuracy: {test_accuracy * 100:.2f}%\n")

print(f"Training results saved to: {results_path}")