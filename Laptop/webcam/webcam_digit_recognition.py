import cv2
import numpy as np
import tensorflow as tf
import os
import csv
from datetime import datetime


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

FP32_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "mnist_cnn_fp32.tflite"
)

INT8_MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "models",
    "mnist_cnn_int8.tflite"
)

EVALUATION_FOLDER = os.path.join(
    PROJECT_ROOT,
    "evaluation"
)

CSV_PATH = os.path.join(
    EVALUATION_FOLDER,
    "webcam_results.csv"
)

SCREENSHOT_FOLDER = os.path.join(
    EVALUATION_FOLDER,
    "webcam_screenshots"
)

os.makedirs(
    EVALUATION_FOLDER,
    exist_ok=True
)

os.makedirs(
    SCREENSHOT_FOLDER,
    exist_ok=True
)


# ============================================================
# CHECK MODELS
# ============================================================

if not os.path.exists(FP32_MODEL_PATH):
    raise FileNotFoundError(
        "FP32 model not found:\n"
        + FP32_MODEL_PATH
    )

if not os.path.exists(INT8_MODEL_PATH):
    raise FileNotFoundError(
        "INT8 model not found:\n"
        + INT8_MODEL_PATH
    )


# ============================================================
# LOAD FP32 MODEL
# ============================================================

print("Loading FP32 model...")

fp32_interpreter = tf.lite.Interpreter(
    model_path=FP32_MODEL_PATH
)

fp32_interpreter.allocate_tensors()

fp32_input = (
    fp32_interpreter
    .get_input_details()[0]
)

fp32_output = (
    fp32_interpreter
    .get_output_details()[0]
)

print(
    "FP32 input shape:",
    fp32_input["shape"]
)

print(
    "FP32 input type:",
    fp32_input["dtype"]
)


# ============================================================
# LOAD INT8 MODEL
# ============================================================

print("Loading INT8 model...")

int8_interpreter = tf.lite.Interpreter(
    model_path=INT8_MODEL_PATH
)

int8_interpreter.allocate_tensors()

int8_input = (
    int8_interpreter
    .get_input_details()[0]
)

int8_output = (
    int8_interpreter
    .get_output_details()[0]
)

print(
    "INT8 input shape:",
    int8_input["shape"]
)

print(
    "INT8 input type:",
    int8_input["dtype"]
)

print("Models loaded successfully.")


# ============================================================
# CREATE CSV
# ============================================================

if not os.path.exists(CSV_PATH):

    with open(
        CSV_PATH,
        "w",
        newline=""
    ) as file:

        writer = csv.writer(file)

        writer.writerow([
            "test_number",
            "actual_digit",
            "fp32_prediction",
            "int8_prediction",
            "fp32_confidence_percent",
            "int8_confidence_percent",
            "fp32_correct",
            "int8_correct",
            "timestamp",
            "screenshot"
        ])

    print(
        "Created CSV:"
    )

    print(
        CSV_PATH
    )


# ============================================================
# FIND NEXT TEST NUMBER
# ============================================================

with open(
    CSV_PATH,
    "r",
    newline=""
) as file:

    rows = list(
        csv.reader(file)
    )

# Header = 1 row
# Therefore:
# 1 row  -> test 1
# 2 rows -> test 2
# etc.

test_number = len(rows)


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_digit(frame):

    """
    Automatically detect a handwritten digit anywhere
    in the webcam frame.

    Final model input:
        1 x 28 x 28 x 1
        grayscale
        white digit on black background
        pixel values 0-1
    """

    # --------------------------------------------------------
    # Convert to grayscale
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )


    # --------------------------------------------------------
    # Reduce camera noise
    # --------------------------------------------------------

    gray = cv2.GaussianBlur(
        gray,
        (5, 5),
        0
    )


    # --------------------------------------------------------
    # Threshold
    #
    # Dark handwriting becomes white.
    # White paper becomes black.
    # --------------------------------------------------------

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV
        + cv2.THRESH_OTSU
    )


    # --------------------------------------------------------
    # Remove small noise
    # --------------------------------------------------------

    kernel = np.ones(
        (3, 3),
        np.uint8
    )

    binary = cv2.morphologyEx(
        binary,
        cv2.MORPH_OPEN,
        kernel
    )


    # --------------------------------------------------------
    # Find contours
    # --------------------------------------------------------

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    # --------------------------------------------------------
    # Blank MNIST image
    # --------------------------------------------------------

    canvas = np.zeros(
        (28, 28),
        dtype=np.uint8
    )


    # --------------------------------------------------------
    # No contours
    # --------------------------------------------------------

    if not contours:

        image = canvas.astype(
            np.float32
        ) / 255.0

        image = image.reshape(
            1,
            28,
            28,
            1
        )

        return (
            image,
            None,
            binary
        )


    # --------------------------------------------------------
    # Frame dimensions
    # --------------------------------------------------------

    frame_height, frame_width = (
        gray.shape
    )

    frame_area = (
        frame_height *
        frame_width
    )

    candidates = []


    # --------------------------------------------------------
    # Find meaningful contours
    # --------------------------------------------------------

    for contour in contours:

        area = cv2.contourArea(
            contour
        )

        # Ignore tiny noise
        if area < 300:
            continue

        x, y, w, h = (
            cv2.boundingRect(
                contour
            )
        )

        # Ignore objects covering most
        # of the frame
        if (w * h) > frame_area * 0.5:
            continue

        # Ignore extremely small dimensions
        if w < 10 or h < 10:
            continue

        candidates.append(
            (
                contour,
                area,
                x,
                y,
                w,
                h
            )
        )


    # --------------------------------------------------------
    # No meaningful digit found
    # --------------------------------------------------------

    if not candidates:

        image = canvas.astype(
            np.float32
        ) / 255.0

        image = image.reshape(
            1,
            28,
            28,
            1
        )

        return (
            image,
            None,
            binary
        )


    # --------------------------------------------------------
    # Select largest meaningful contour
    # --------------------------------------------------------

    (
        contour,
        area,
        x,
        y,
        w,
        h
    ) = max(
        candidates,
        key=lambda item: item[1]
    )


    # --------------------------------------------------------
    # Add padding
    #
    # KEEPING THE PREVIOUS VERSION'S METHOD
    # --------------------------------------------------------

    padding = int(
        0.20 *
        max(w, h)
    )

    x_start = max(
        0,
        x - padding
    )

    y_start = max(
        0,
        y - padding
    )

    x_end = min(
        frame_width,
        x + w + padding
    )

    y_end = min(
        frame_height,
        y + h + padding
    )


    # --------------------------------------------------------
    # Crop digit
    # --------------------------------------------------------

    digit = binary[
        y_start:y_end,
        x_start:x_end
    ]


    # --------------------------------------------------------
    # Preserve aspect ratio
    #
    # THIS IS THE PREVIOUS VERSION'S METHOD
    # --------------------------------------------------------

    digit_height, digit_width = (
        digit.shape
    )

    scale = (
        20.0 /
        max(
            digit_height,
            digit_width
        )
    )

    new_width = max(
        1,
        int(
            round(
                digit_width *
                scale
            )
        )
    )

    new_height = max(
        1,
        int(
            round(
                digit_height *
                scale
            )
        )
    )


    digit_resized = cv2.resize(
        digit,
        (
            new_width,
            new_height
        ),
        interpolation=cv2.INTER_AREA
    )


    # --------------------------------------------------------
    # Center digit in 28x28
    # --------------------------------------------------------

    start_x = (
        28 -
        new_width
    ) // 2

    start_y = (
        28 -
        new_height
    ) // 2


    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if (
        start_x < 0
        or start_y < 0
        or start_x + new_width > 28
        or start_y + new_height > 28
    ):

        digit_resized = cv2.resize(
            digit,
            (20, 20),
            interpolation=cv2.INTER_AREA
        )

        new_width = 20
        new_height = 20

        start_x = 4
        start_y = 4


    # --------------------------------------------------------
    # Place digit on canvas
    # --------------------------------------------------------

    canvas[
        start_y:start_y + new_height,
        start_x:start_x + new_width
    ] = digit_resized


    # --------------------------------------------------------
    # Normalize exactly as training
    # --------------------------------------------------------

    image = (
        canvas.astype(
            np.float32
        ) / 255.0
    )


    # --------------------------------------------------------
    # FINAL SHAPE
    #
    # (28,28)
    #     ↓
    # (1,28,28,1)
    # --------------------------------------------------------

    image = image.reshape(
        1,
        28,
        28,
        1
    )


    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    bounding_box = (
        x_start,
        y_start,
        x_end,
        y_end
    )


    return (
        image,
        bounding_box,
        binary
    )


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_fp32_input(image):

    """
    Guarantees FP32 input shape:
        (1, 28, 28, 1)
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    # Remove unnecessary dimensions
    image = np.squeeze(image)

    # Now image should be 28x28
    if image.shape != (28, 28):

        raise ValueError(
            "Unexpected image shape before FP32 "
            "conversion: "
            + str(image.shape)
        )

    # Add channel
    image = np.expand_dims(
        image,
        axis=-1
    )

    # Add batch
    image = np.expand_dims(
        image,
        axis=0
    )

    return image.astype(
        np.float32
    )


# ============================================================
# PREPARE INT8 INPUT
# ============================================================

def prepare_int8_input(image):

    """
    Guarantees INT8 input shape:
        (1, 28, 28, 1)
    """

    image = np.asarray(
        image,
        dtype=np.float32
    )

    # Remove unnecessary dimensions
    image = np.squeeze(image)

    # Now image should be 28x28
    if image.shape != (28, 28):

        raise ValueError(
            "Unexpected image shape before INT8 "
            "conversion: "
            + str(image.shape)
        )

    # Get quantization parameters
    scale, zero_point = (
        int8_input["quantization"]
    )

    # Float -> INT8
    quantized = (
        image / scale
    ) + zero_point

    quantized = np.round(
        quantized
    )

    quantized = np.clip(
        quantized,
        -128,
        127
    )

    quantized = quantized.astype(
        np.int8
    )

    # Add channel
    quantized = np.expand_dims(
        quantized,
        axis=-1
    )

    # Add batch
    quantized = np.expand_dims(
        quantized,
        axis=0
    )

    return quantized


# ============================================================
# FP32 PREDICTION
# ============================================================

def predict_fp32(image):

    # Prepare guaranteed 4D input
    model_input = prepare_fp32_input(
        image
    )

    # Final safety check
    if model_input.shape != (
        1,
        28,
        28,
        1
    ):

        raise ValueError(
            "FP32 model input shape is "
            + str(model_input.shape)
        )

    fp32_interpreter.set_tensor(
        fp32_input["index"],
        model_input
    )

    fp32_interpreter.invoke()

    output = (
        fp32_interpreter
        .get_tensor(
            fp32_output["index"]
        )[0]
    )

    digit = int(
        np.argmax(output)
    )

    confidence = float(
        np.max(output)
    )

    return (
        digit,
        confidence
    )


# ============================================================
# INT8 PREDICTION
# ============================================================

def predict_int8(image):

    # Prepare guaranteed 4D INT8 input
    model_input = prepare_int8_input(
        image
    )

    # Final safety check
    if model_input.shape != (
        1,
        28,
        28,
        1
    ):

        raise ValueError(
            "INT8 model input shape is "
            + str(model_input.shape)
        )

    int8_interpreter.set_tensor(
        int8_input["index"],
        model_input
    )

    int8_interpreter.invoke()

    output = (
        int8_interpreter
        .get_tensor(
            int8_output["index"]
        )[0]
    )

    # INT8 -> float
    output_scale, output_zero_point = (
        int8_output["quantization"]
    )

    output_float = (
        output.astype(
            np.float32
        )
        - output_zero_point
    ) * output_scale

    digit = int(
        np.argmax(output_float)
    )

    confidence = float(
        np.max(output_float)
    )

    return (
        digit,
        confidence
    )


# ============================================================
# OPEN WEBCAM
# ============================================================

print(
    "\nOpening laptop webcam..."
)

camera = cv2.VideoCapture(0)

if not camera.isOpened():

    print(
        "ERROR: Could not open webcam."
    )

    print(
        "Try changing VideoCapture(0)"
    )

    print(
        "to VideoCapture(1)."
    )

    exit()


print(
    "Webcam opened successfully."
)


# ============================================================
# INSTRUCTIONS
# ============================================================

print("\n========================================")
print("MNIST EDGE AI WEBCAM RECOGNITION")
print("========================================")
print("Hold ONE digit anywhere in view.")
print("")
print("Press the ACTUAL digit key to SAVE:")
print("")
print("  0 -> save actual digit 0")
print("  1 -> save actual digit 1")
print("  2 -> save actual digit 2")
print("  3 -> save actual digit 3")
print("  4 -> save actual digit 4")
print("  5 -> save actual digit 5")
print("  6 -> save actual digit 6")
print("  7 -> save actual digit 7")
print("  8 -> save actual digit 8")
print("  9 -> save actual digit 9")
print("")
print("Q = Quit")
print("========================================\n")


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = camera.read()

    if not ret:

        print(
            "ERROR: Could not read webcam frame."
        )

        break


    # --------------------------------------------------------
    # NO HORIZONTAL FLIP
    # --------------------------------------------------------

    height, width = (
        frame.shape[:2]
    )


    # --------------------------------------------------------
    # Detect and preprocess digit
    # --------------------------------------------------------

    (
        digit_image,
        bounding_box,
        binary
    ) = preprocess_digit(
        frame
    )


    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    fp32_digit, fp32_confidence = (
        predict_fp32(
            digit_image
        )
    )

    int8_digit, int8_confidence = (
        predict_int8(
            digit_image
        )
    )


    # --------------------------------------------------------
    # Draw bounding box
    # --------------------------------------------------------

    if bounding_box is not None:

        bx1, by1, bx2, by2 = (
            bounding_box
        )

        cv2.rectangle(
            frame,
            (bx1, by1),
            (bx2, by2),
            (0, 255, 0),
            3
        )


    # --------------------------------------------------------
    # Display FP32 prediction
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"FP32: {fp32_digit} "
        f"({fp32_confidence * 100:.1f}%)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Display INT8 prediction
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"INT8: {int8_digit} "
        f"({int8_confidence * 100:.1f}%)",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Display test number
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Test: {min(test_number, 10)}/10",
        (20, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Instructions
    # --------------------------------------------------------

    cv2.putText(
        frame,
        "Press actual digit 0-9 to SAVE",
        (20, height - 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "Q = QUIT",
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2
    )


    # --------------------------------------------------------
    # Show webcam
    # --------------------------------------------------------

    cv2.imshow(
        "MNIST Edge AI - Webcam",
        frame
    )


    # --------------------------------------------------------
    # Show processed MNIST input
    # --------------------------------------------------------

    processed_display = (
        digit_image[0, :, :, 0] * 255
    ).astype(
        np.uint8
    )

    processed_display = cv2.resize(
        processed_display,
        (280, 280),
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imshow(
        "Processed MNIST Input",
        processed_display
    )


    # --------------------------------------------------------
    # Keyboard
    # --------------------------------------------------------

    key = cv2.waitKey(1) & 0xFF


    # ========================================================
    # SAVE USING ACTUAL DIGIT KEY
    # ========================================================

    if key in [
        ord("0"),
        ord("1"),
        ord("2"),
        ord("3"),
        ord("4"),
        ord("5"),
        ord("6"),
        ord("7"),
        ord("8"),
        ord("9")
    ]:

        # ----------------------------------------------------
        # Check test limit
        # ----------------------------------------------------

        if test_number > 10:

            print(
                "\n10 tests already completed."
            )

            continue


        # ----------------------------------------------------
        # Actual digit = key pressed
        # ----------------------------------------------------

        actual_digit = int(
            chr(key)
        )


        print("\n========================================")

        print(
            f"SAVING TEST #{test_number}"
        )

        print(
            f"Actual digit       : {actual_digit}"
        )

        print(
            f"FP32 prediction    : {fp32_digit}"
        )

        print(
            f"INT8 prediction    : {int8_digit}"
        )


        # ----------------------------------------------------
        # Correctness
        # ----------------------------------------------------

        fp32_correct = (
            fp32_digit ==
            actual_digit
        )

        int8_correct = (
            int8_digit ==
            actual_digit
        )


        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp = datetime.now().strftime(
            "%Y-%m-%d_%H-%M-%S"
        )


        # ----------------------------------------------------
        # Screenshot filename
        # ----------------------------------------------------

        screenshot_filename = (
            f"test_{test_number:02d}_"
            f"digit_{actual_digit}_"
            f"{timestamp}.jpg"
        )


        screenshot_path = os.path.join(
            SCREENSHOT_FOLDER,
            screenshot_filename
        )


        # ----------------------------------------------------
        # SAVE SCREENSHOT
        # ----------------------------------------------------

        screenshot_saved = cv2.imwrite(
            screenshot_path,
            frame
        )


        # ----------------------------------------------------
        # SAVE CSV
        # ----------------------------------------------------

        with open(
            CSV_PATH,
            "a",
            newline=""
        ) as file:

            writer = csv.writer(
                file
            )

            writer.writerow([
                test_number,
                actual_digit,
                fp32_digit,
                int8_digit,
                round(
                    fp32_confidence * 100,
                    2
                ),
                round(
                    int8_confidence * 100,
                    2
                ),
                (
                    "Yes"
                    if fp32_correct
                    else "No"
                ),
                (
                    "Yes"
                    if int8_correct
                    else "No"
                ),
                timestamp,
                screenshot_filename
            ])


        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print(
            f"FP32 confidence : "
            f"{fp32_confidence * 100:.2f}%"
        )

        print(
            f"INT8 confidence : "
            f"{int8_confidence * 100:.2f}%"
        )

        print(
            "FP32 correct    : "
            + (
                "Yes"
                if fp32_correct
                else "No"
            )
        )

        print(
            "INT8 correct    : "
            + (
                "Yes"
                if int8_correct
                else "No"
            )
        )


        if screenshot_saved:

            print(
                "Screenshot saved successfully."
            )

            print(
                screenshot_path
            )

        else:

            print(
                "WARNING: Screenshot could not be saved."
            )


        print(
            "\nCSV updated:"
        )

        print(
            CSV_PATH
        )

        print(
            "========================================"
        )


        # ----------------------------------------------------
        # Next test
        # ----------------------------------------------------

        test_number += 1


        if test_number > 10:

            print("\n========================================")
            print("ALL 10 WEBCAM TESTS COMPLETED")
            print("========================================")

            print(
                "Results saved to:"
            )

            print(
                CSV_PATH
            )

            print(
                "\nScreenshots saved to:"
            )

            print(
                SCREENSHOT_FOLDER
            )

            print(
                "========================================\n"
            )


    # ========================================================
    # Q = QUIT
    # ========================================================

    elif key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

camera.release()

cv2.destroyAllWindows()

print("\n========================================")
print("WEBCAM TEST FINISHED")
print("========================================")

print(
    "CSV file:"
)

print(
    CSV_PATH
)

print(
    "\nScreenshots:"
)

print(
    SCREENSHOT_FOLDER
)