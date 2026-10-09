# MNIST Edge AI — Digit Recognition on Laptop, Raspberry Pi and Pico

A lightweight handwritten digit recognition project using the MNIST dataset, TensorFlow/Keras, TensorFlow Lite, and edge deployment.

The project explores digit classification, full-integer quantization, webcam-based image capture, and local inference on edge hardware.

**Institution:** Amrita Vishwa Vidyapeetham, Amrita School of Artificial Intelligence  
**Subject:** Introduction to Electronics and Elements of Computing  
**Team:** 13

## Team Members

| Name | Registration Number |
|---|---|
| Janak Dhillon | CB.AI.U4AIM24016 |
| M R Bharath | CB.AI.U4AIM24022 |
| Srisha Satish Kanna | CB.AI.U4AIM24046 |
| Abhinav Sarvesh D | CB.AI.U4AIM24052 |

## Project Overview

The objective is to build a digit recognition system that captures handwritten or displayed digits, preprocesses the image, and predicts the digit from 0 to 9 using a trained neural network.

The repository contains three project components:

- **Laptop:** Model training, conversion, webcam capture, and communication with edge devices.
- **Raspberry Pi:** Local inference using TensorFlow Lite FP32 and INT8 models, plus performance benchmarking.
- **Pico:** The project's microcontroller implementation and associated source files.

The system demonstrates how a trained machine learning model can be optimized and deployed beyond a desktop environment.

## Repository Structure

```text
mnist_edge_ai/
├── Laptop/
│   ├── models/
│   │   ├── mnist_cnn_fp32.keras
│   │   ├── mnist_cnn_fp32.tflite
│   │   ├── mnist_cnn_int8.tflite
│   │   └── mnist_model_data.cpp
│   ├── webcam/
│   │   └── webcam_digit_recognition.py
│   └── ...
├── pico/
│   └── Pico source files and deployment assets
├── benchmark_pi.py
├── pi_inference_server.py
├── pi_inference_server_fp32.py
├── test_pi_model.py
├── LICENSE
└── README.md
```

*Note: This tree highlights the known project files. Other files may be present in the repository.*

## Model Architecture

A lightweight Convolutional Neural Network (CNN) was trained on the MNIST handwritten digit dataset.

| Layer | Configuration |
|---|---|
| Input | 28 × 28 × 1 grayscale image |
| Convolution 1 | 16 filters, 3 × 3, ReLU |
| Max pooling 1 | 2 × 2 |
| Convolution 2 | 32 filters, 3 × 3, ReLU |
| Max pooling 2 | 2 × 2 |
| Flatten | Convert feature maps into a vector |
| Dense | 32 units, ReLU |
| Output | 10 units, Softmax |

**Training configuration**

- Dataset: MNIST
- Training samples: 54,000
- Validation samples: 6,000
- Test samples: 10,000
- Random seed: 42
- Epochs: 10
- Batch size: 128
- Optimizer: Adam
- Loss: Sparse categorical cross-entropy
- Trainable parameters: 30,762
- Input normalization: pixel values scaled to [0, 1]

## Model Optimization and Quantization

The trained Keras model was converted into TensorFlow Lite formats for edge deployment.

### FP32 model

The FP32 model retains 32-bit floating-point weights and activations.

File: `Laptop/models/mnist_cnn_fp32.tflite`

### Full-integer INT8 model

The INT8 model was generated using TensorFlow Lite post-training quantization with a representative dataset of 500 training images.

The conversion uses integer-only built-in TensorFlow Lite operations and INT8 input/output tensors.

File: `Laptop/models/mnist_cnn_int8.tflite`

Quantization reduces storage requirements and can improve inference efficiency on supported hardware.

## Laptop Results

The following measurements were obtained during the laptop model evaluation.

| Metric | FP32 | INT8 |
|---|---:|---:|
| MNIST test accuracy | 98.76% | 98.74% |
| Model size | 126,452 bytes | 36,128 bytes |
| Mean inference latency | 0.0523 ms | 0.0451 ms |
| Median inference latency | 0.0517 ms | 0.0406 ms |
| P95 inference latency | 0.0645 ms | 0.0656 ms |
| Approximate RSS increase | 0.93 MB | 0.04 MB |

The INT8 model reduced the model file size by approximately **71.43%**, with an accuracy decrease of only 0.02 percentage points.

The laptop measurements were performed using the TensorFlow Lite interpreter with one thread, 20 warm-up runs, and 1,000 timed inferences.

## Raspberry Pi Deployment

The Raspberry Pi 3B+ runs the TensorFlow Lite model locally using `tflite-runtime`.

Two inference servers are provided:

- `pi_inference_server.py` — INT8 model inference.
- `pi_inference_server_fp32.py` — FP32 model inference.

The model is loaded on the Pi, and inference takes place locally rather than through a cloud prediction service.

### Laptop-to-Pi communication

The laptop and Raspberry Pi communicate through a local network using TCP sockets.

1. The laptop webcam captures a digit.
2. OpenCV preprocesses the image into a 28 × 28 grayscale image.
3. The processed INT8 pixel values are sent to the Raspberry Pi over TCP.
4. The Pi runs its selected TensorFlow Lite model.
5. The Pi returns a JSON response containing the predicted digit, confidence, and inference latency.

The development setup used TCP port `5000`. The Pi's IP address can change, so configure the current address in the laptop client before running the system.

### Raspberry Pi benchmark results

These measurements were obtained on the Raspberry Pi 3B+ using `benchmark_pi.py`, with one interpreter thread, 20 warm-up inferences, and 500 timed inference runs.

| Metric | FP32 | INT8 |
|---|---:|---:|
| Model size | 126,452 bytes | 36,128 bytes |
| Input/output type | float32 | int8 |
| Mean inference latency | 1.004 ms | 0.799 ms |
| Median inference latency | 0.996 ms | 0.795 ms |
| P95 inference latency | 1.055 ms | 0.812 ms |
| Approximate RSS increase | 2.31 MiB | 2.22 MiB |

On the Raspberry Pi, INT8 quantization reduced model size by approximately **71.43%** and mean inference latency by approximately **20.42%** compared with FP32.

The RSS figures represent approximate process memory changes after interpreter allocation; they are not exact measurements of model-only memory consumption.

### Raspberry Pi setup

The recorded development environment used Python 3.10.13, NumPy 1.26.4, and `tflite-runtime` 2.14.0.

From the cloned repository, activate the existing virtual environment:

```bash
cd ~/mnist_edge_ai
source ~/mnist-pi-venv/bin/activate
```

Run the INT8 inference server:

```bash
python pi_inference_server.py
```

For FP32 inference, stop the first server and run:

```bash
python pi_inference_server_fp32.py
```

The two servers use the same TCP port, so run only one at a time.

Run the standalone INT8 model test:

```bash
python test_pi_model.py
```

Run the INT8 benchmark:

```bash
python benchmark_pi.py Laptop/models/mnist_cnn_int8.tflite
```

Run the FP32 benchmark:

```bash
python benchmark_pi.py Laptop/models/mnist_cnn_fp32.tflite
```

## Laptop Webcam

The webcam implementation is located in:

`Laptop/webcam/webcam_digit_recognition.py`

The webcam workflow includes:

1. Frame capture using OpenCV.
2. Grayscale conversion and Gaussian blur.
3. Otsu thresholding and morphological opening.
4. Contour detection and digit-region selection.
5. Cropping, padding, aspect-ratio preservation, and resizing.
6. Conversion to a 28 × 28 model input.
7. Sending the processed image to the Raspberry Pi for prediction.

For the TCP deployment client, ensure the Pi server is running and the IP address and port in the laptop script match the Pi's current network configuration.

## Pico Component

The `pico/` directory contains the microcontroller portion of the project.

The Pico component is included as a separate target so that its implementation and deployment assets remain organized independently from the laptop and Raspberry Pi software.

Refer to the source files in `pico/` for its specific firmware, build requirements, and execution instructions. The exact Pico hardware configuration and inference implementation should be documented according to the files present in that directory.

## Real-World Webcam Testing

Physical webcam trials were performed using handwritten or displayed digits.

The experiments included examples from all ten digit classes, 0 through 9. Some digits were recognized correctly on the first attempt, while others required repeated attempts and were occasionally misclassified.

These webcam trials are distinct from the formal MNIST test-set evaluation. The MNIST test accuracy should not be interpreted as the accuracy of arbitrary real-world webcam inputs.

## Performance Comparison

The measured results show three important outcomes:

1. **Storage efficiency:** INT8 reduces the model file size by approximately 71.43%.
2. **Inference performance:** On the Raspberry Pi benchmark, INT8 mean inference latency was 0.799 ms versus 1.004 ms for FP32.
3. **Accuracy retention:** The laptop MNIST test accuracy changed from 98.76% for FP32 to 98.74% for INT8.

The benchmark measures the model's inference call, not the total webcam-to-prediction time. Network transfer, image preprocessing, and application overhead are separate costs.

## Technologies Used

- Python
- TensorFlow and Keras
- TensorFlow Lite
- NumPy
- OpenCV
- TCP sockets
- Raspberry Pi 3B+
- Raspberry Pi OS
- Raspberry Pi Pico project files

## Future Improvements

- Collect a controlled, fixed set of webcam images for a fair physical-digit accuracy comparison.
- Improve preprocessing for different handwriting styles, lighting conditions, and backgrounds.
- Measure complete end-to-end latency, including image capture, preprocessing, network transfer, and inference.
- Record memory usage using a consistent methodology across target devices.
- Document and validate the exact Pico deployment procedure based on its source files.

## License

See the repository's `LICENSE` file for licensing terms.
