# Copyright 2021 ETH Zurich and University of Bologna.
# Licensed under the Apache License, Version 2.0, see https://www.apache.org/licenses/LICENSE-2.0 for details.
# SPDX-License-Identifier: Apache-2.0
#
# Author: Viviane Potocnik <vivianep@iis.ee.ethz.ch> (ETH Zurich)


import matplotlib

matplotlib.use("Agg")

import time

import customtkinter as ctk
import cv2
import matplotlib.pyplot as plt
import numpy as np
import serial

img_size = (32, 32)

"""Runs a live demonstration of a depth estimation model running on 
a microcontroller (MCU). This function connects to the MCU over a 
serial (USB) cable, sends in test input images to the depth estimation
model running on an MCU, receives the predicted depth image, and 
graphs a comparison between the depth prediction and the actual image."""
def main(x_test, y_test, ser: serial.Serial):
    # define how many images from the test set to send to the MCU
    test_len = 2
    num_pixels_in = np.product(img_size) * 1
    num_pixels_out = np.product(img_size) * 1

    # Get a depth estimation from the model running on the MCU 
    # using an example test image
    _ = get_pred(ser, img_size, num_pixels_in, x_test[0], num_pixels_out=num_pixels_out)
    time.sleep(2)

    # Reset the buffers on the USB cable
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # Repeat the above steps for all images in the test dataset
    for req_img, true_depth in zip(x_test[:test_len], y_test[:test_len]):
        req_img, pred = get_pred(
            ser, img_size, num_pixels_in, req_img, num_pixels_out=num_pixels_out
        )
        print(pred.shape)
        pred = pred.astype("float32") / 255
        mae = np.mean(np.abs(true_depth - pred))
        print(f"MAE: {mae}")

        ser.reset_input_buffer()
        ser.reset_output_buffer()

"""Sends an input image to the depth estimation model running on the MCU and receives
a depth prediction over a USB cable. The predicted depth image is then displayed side
by side with the provided input image."""
def get_pred(ser, img_size, num_pixels_in, req_img, num_pixels_out=None):
    if num_pixels_out is None:
        num_pixels_out = num_pixels_in
    
    # Resize the image to the model's input if it is not 
    # already dized properly
    req_img = cv2.resize(req_img, img_size)
    # Normalize the image and convert it to unsigned 8 bit format
    req_img = (req_img * 255).astype("uint8")
    # Send the input image to the depth estimation model over the USB cable
    ser.write(req_img.tobytes())
    ser.flush()
    # Set up variables for reading in the input image 
    received_bytes = 0
    data = b""
    expected_bytes = num_pixels_in
    start = time.time()
    # Read in the repeated input image chunk-by-chunk \
    # from the serial cable until it is complete
    while received_bytes < expected_bytes:
        remaining_bytes = expected_bytes - received_bytes
        chunk = ser.read(remaining_bytes)
        received_bytes += len(chunk)
        print(f"{received_bytes=}")
        data += chunk
    print(f"Received {received_bytes} bytes in {time.time() - start:.2f} seconds")
    # Reconstruct the repeated detph image 
    resp_img = np.frombuffer(data, dtype=np.uint8)
    # Assert that the re[eated] image was reconstructed properly
    assert (
        len(resp_img) == num_pixels_in
    ), f"Expected {num_pixels_in} bytes, got {len(resp_img)}"
    # Sleep for 5 seconds to prevent USB thrashing
    time.sleep(5)
    # Read in the predicted depth image from the serial cable 
    # and reconstruct it 
    pred_bytes = ser.read(num_pixels_out)
    pred = np.frombuffer(pred_bytes, dtype=np.uint8)
    # Plot the input image, repeated input image from the model, 
    # and the depth estimation image next to each other
    fig, axs = plt.subplots(1, 3, figsize=(10, 5))
    axs[0].imshow(req_img)
    axs[1].imshow(resp_img.reshape(img_size))
    axs[2].imshow(pred.reshape(img_size))

    axs[0].set_title("Request Image")
    axs[1].set_title("Response Image")
    axs[2].set_title("Depth Prediction")

    # Save the generated figure and return the input image
    # and predicted depth image
    plt.savefig("prediction.png")
    pred = pred.reshape(img_size)
    req_img = req_img.reshape(img_size)
    return req_img, pred

"""Runs the live example code and reads in command line arguments
before running."""
if __name__ == "__main__":
    import argparse

    # Get the dataset location from the user through the CLI
    parser = argparse.ArgumentParser()
    parser.add_argument("--ds_name", type=str, required=False, default="fmnist")
    args, _ = parser.parse_known_args()
    print(args)

    # Load example test depth images from the disk
    x_test = np.load("test_data/x_test_depth_32.npy")
    y_test = np.load("test_data/y_test_depth_32.npy").squeeze()

    ctk.set_appearance_mode("dark")
    # ctk.set_default_color_theme("dark-blue")

    print(f"Loaded x with shape: {x_test.shape}")
    print(f"Loaded y with shape: {y_test.shape}")

    # Open and clear the USB port 
    serial_port = "/dev/ttyACM0"
    ser = serial.Serial(port=serial_port, baudrate=115200, timeout=20)
    # flush the serial port
    ser.flush()
    ser.flushInput()
    ser.flushOutput()
    ser.reset_input_buffer()
    ser.reset_output_buffer()

    # Run the example and close the serial connection if anything
    # errors out
    try:
        main(x_test, y_test, ser)
    finally:
        ser.close()