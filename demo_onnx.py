#!/usr/bin/env python
# -*- coding: utf-8 -*-
import copy
import time
import argparse

import cv2 as cv
import numpy as np
import onnxruntime

import cvui


def run_inference(onnx_session, image, gamma, strength):
    # ONNX Infomation
    input_size = onnx_session.get_inputs()[0].shape
    input_width = input_size[3]
    input_height = input_size[2]

    # Pre process:Resize, BGR->RGB, Transpose, float32 cast
    input_image = cv.resize(image, dsize=(input_width, input_height))
    input_image = cv.cvtColor(input_image, cv.COLOR_BGR2RGB)
    input_image = input_image.transpose(2, 0, 1)
    input_image = np.expand_dims(input_image, axis=0)
    input_image = input_image.astype('float32')
    input_image = input_image / 255.0

    # Inference
    input_name_image = onnx_session.get_inputs()[0].name
    input_name_gamma = onnx_session.get_inputs()[1].name
    input_name_strength = onnx_session.get_inputs()[2].name
    result = onnx_session.run(
        None,
        {
            input_name_image: input_image,
            input_name_gamma: np.array(gamma).astype(np.double),
            input_name_strength: np.array(strength).astype(np.double)
        },
    )

    # Post process:squeeze, RGB->BGR, Transpose, uint8 cast
    output_image = np.squeeze(result[0])
    output_image = output_image.transpose(1, 2, 0)
    output_image = np.clip(output_image * 255.0, 0, 255)
    output_image = output_image.astype(np.uint8)
    output_image = cv.cvtColor(output_image, cv.COLOR_RGB2BGR)

    return output_image


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--movie", type=str, default=None)
    parser.add_argument(
        "--model",
        type=str,
        default='model/Bread_320x240.onnx',
    )

    args = parser.parse_args()
    model_path = args.model

    # Initialize video capture
    cap_device = args.device
    if args.movie is not None:
        cap_device = args.movie
    cap = cv.VideoCapture(cap_device)

    # Load model
    onnx_session = onnxruntime.InferenceSession(
        model_path,
        providers=[
            'CUDAExecutionProvider',
            'CPUExecutionProvider',
        ],
    )

    # Parameter Window
    cvui.init('Bread Parameters')
    gamma_value = [1.0]
    strength_value = [0.05]
    parameter_window = np.zeros((160, 320, 3), np.uint8)

    while True:
        # Parameter Window Update
        parameter_window[:] = (49, 52, 49)

        cvui.text(parameter_window, 10, 10, 'Gamma')
        cvui.trackbar(parameter_window, 10, 30, 300, gamma_value, 0.0, 1.5, 1,
                      '%.1Lf', cvui.TRACKBAR_DISCRETE, 0.1)

        cvui.text(parameter_window, 10, 80, 'Strength')
        cvui.trackbar(parameter_window, 10, 100, 300, strength_value, 0.0, 0.2,
                      1, '%.2Lf', cvui.TRACKBAR_DISCRETE, 0.01)
        cvui.update()

        start_time = time.time()

        # Capture read
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv.resize(frame, dsize=None, fx=0.75, fy=0.75)
        debug_image = copy.deepcopy(frame)
        frame_height, frame_width = frame.shape[0], frame.shape[1]

        # Inference execution
        output_image = run_inference(
            onnx_session,
            frame,
            gamma_value[0],
            strength_value[0],
        )

        output_image = cv.resize(output_image,
                                 dsize=(frame_width, frame_height))

        elapsed_time = time.time() - start_time

        # Inference elapsed time
        cv.putText(
            debug_image,
            "Elapsed Time : " + '{:.1f}'.format(elapsed_time * 1000) + "ms",
            (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1, cv.LINE_AA)

        key = cv.waitKey(1)
        if key == 27:  # ESC
            break
        cv.imshow('Bread Input', debug_image)
        cv.imshow('Bread Output', output_image)
        cv.imshow('Bread Parameters', parameter_window)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
