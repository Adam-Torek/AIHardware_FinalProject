import json
import pathlib

import numpy as np
import tensorflow as tf


# Helper function to run inference on a TFLite model
"""Runs inference testing on a TFLite Model to validate 
that the TFLite-converted model is comparably performant
to its non-compact version."""
def run_tflite_model(tflite_file, test_ds):
    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
    interpreter.allocate_tensors()

    # Get the input and output sizes and annotations of the
    # TFLite file format
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Create lists to store model predictions and metric results 
    predictions = []
    maes = []
    mses = []

    # Check the sample dataset to make sure its format is
    # compatible with the tester 
    sample = next(iter(test_ds))
    if len(sample[0].shape) == 3:
        ds = test_ds
    else:
        ds = test_ds.unbatch().batch(1)

    # Evaluation loop to test the TFLite model on the 
    # evaluation depth dataset
    for i, (test_image, test_depth) in enumerate(ds):
        if i % 100 == 0 and i > 0:
            print("Evaluated on %d images." % i)

        # Check if the input type is quantized, then rescale input data to uint8
        if input_details["dtype"] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            test_image = test_image / input_scale + input_zero_point

        # Get the test depth image from the test dataset 
        if len(test_image.shape) == 3:
            test_image = np.expand_dims(test_image, axis=0)
        
        # Convert the test image to numpy format
        if not isinstance(test_image, np.ndarray):
            test_image = test_image.numpy()
        test_image = test_image.astype(input_details["dtype"])

        # Send the test image into the TFLite model and extract its
        # results
        interpreter.set_tensor(input_details["index"], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]

        # Get the mean squared error and mean absolute error between
        # the predicted and test depth image pixels
        mse = np.mean((test_depth - output) ** 2)
        mae = np.mean(np.abs(test_depth - output))
        maes.append(mae)
        mses.append(mse)
        predictions.append(output)

    # Return the TFLite model prediction images, mean squared error,  
    # mean absolute error, and metrics in a dictionary format.
    metrics = {
        "mse": np.mean(mses),
        "mae": np.mean(maes),
    }
    return predictions, metrics

"""Evalaute a quantized depth estimation model in TFLite format by
loading the image from disk and sending it into the run_tflite_eval function."""
def eval_quantized_model_in_tflite(tflite_path, test_ds):
    tflite_model_quant_int8_file = pathlib.Path(tflite_path)

    predictions, metrics = run_tflite_model(tflite_model_quant_int8_file, test_ds)

    return metrics

"""Evaluate the given depth estimation model on the provided NYU2 depth estimation
dataset. This function can either evaluate a compiled Keras model or a saved
model in TFLite format. """
def eval_model(
    test_ds, model_name, model=None, tflite_path=None, metrics_file_path=None
):
    # Force either the TFLite model file or the compiled Keras model to 
    # be defined 
    metrics = {}
    assert (
        model is not None or tflite_path is not None
    ), "model or tflite_path must be provided"

    # Load the given TFLite model if the path is provided
    if tflite_path is not None:
        metrics = eval_quantized_model_in_tflite(
            test_ds=test_ds, tflite_path=tflite_path
        )

    # Evaluate the provided TFLite or compiled Keras model on the given 
    # test depth estimation dataset
    if model is not None:
        loss, main_metric = model.evaluate(test_ds, verbose=0)
    else:
        loss = -1
        main_metric = -1
    
    # Get the eval loss and metrics for the model's evaluation
    metrics["eval_loss"] = loss
    metrics["eval_metric"] = main_metric
    print(f"{model_name} model metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.3f}")
        metrics[k] = str(round(v, 3))

    # Load the evaluation metric results from a file if a file path is provided
    if metrics_file_path is not None:
        if pathlib.Path(metrics_file_path).exists():
            # Try to load other metric results from the file path if provided
            try:
                other_metrics = json.load(open(metrics_file_path, "r"))
            # Ignore the metric file if it cannot be opened or parsed
            except json.decoder.JSONDecodeError:
                other_metrics = {}
        else:
            other_metrics = {}
        
        # Load and save the other metrics again 
        # int another JSON file (I don't know why 
        # this is here - seems redundant)
        other_metrics[model_name] = metrics
        json.dump(other_metrics, open(metrics_file_path, "w"))
        print(f"Saved metrics to {metrics_file_path}")
    return metrics