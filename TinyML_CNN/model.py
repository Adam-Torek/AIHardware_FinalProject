import os

import tensorflow as tf
import tensorflow_model_optimization as tfmot
from tensorflow_model_optimization.python.core.keras.compat import keras

"""Function to construct and return the depth estimation CNN model based on
the input image size and number of channels. Note that the depth images are
the same size as the input images but are in grayscale. This constructor function
also includes flags for pruning and quantizing the constructed model using Tensorflow's
model optimization library. This functions upports pruning, sparsity, and quantization-aware
training methods for reducing the model's size and complexity. WARNING: A pruned model must
be passed in pruned_model_unstructured_for_export if both pruning and quantization are turned on.
"""
def get_model(
    img_size,
    in_channels=3,
    use_qat=False,
    use_pruning=False,
    use_pruning_struct=False,
    use_dynamic_sparsity=False,
    pruned_model_unstructured_for_export=None,
    do_reduce_channels=True,
):
    # Create the input layer for the input and depth images
    inputs = keras.layers.Input(shape=(*img_size, in_channels), name="input")

    # Create the filters for downsampling and upsampling
    if do_reduce_channels:
        filters = [32 // 4 * 3, 32 // 4 * 3, 64 // 4 * 3]
    else:
        filters = [32 // 2 * 3, 32 // 2 * 3, 64 // 2 * 3]

    # Set up a baes covolution layer
    x = keras.layers.Conv2D(filters[0], in_channels, strides=2, padding="same")(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(tf.nn.relu)(x)

    previous_block_activation = x  # Set aside residual

    # Downsampling layers to decompose input image using repeatedly
    # smaller and narrow filters in 2D convolution layers
    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filter in filters[1:]:
        x = keras.layers.Activation(tf.nn.relu)(x)
        x = keras.layers.SeparableConv2D(filter, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation(tf.nn.relu)(x)
        x = keras.layers.SeparableConv2D(filter, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = keras.layers.Conv2D(filter, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual


    # Upscale and reconstruct the input image from its decomposition
    # using decovonolutional (Conv2DTranspose) and upsampling layers to get the 
    # depth estimation. 
    ### [Second half of the network: upsampling inputs] ###
    for filter in filters[::-1]:
        x = keras.layers.Activation(tf.nn.relu)(x)
        x = keras.layers.Conv2DTranspose(filter, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.Activation(tf.nn.relu)(x)
        x = keras.layers.Conv2DTranspose(filter, 3, padding="same")(x)
        x = keras.layers.BatchNormalization()(x)

        x = keras.layers.UpSampling2D(2)(x)

        # Project residual deconvolution
        residual = keras.layers.UpSampling2D(2)(previous_block_activation)
        residual = keras.layers.Conv2D(filter, 1, padding="same")(residual)
        x = keras.layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Final convolutional layer to output depth estimation
    outputs = keras.layers.Conv2D(
        1,
        in_channels,
        activation="sigmoid",
        padding="same",
        name="output",
    )(x)

    # Define the model
    model = tf.keras.Model(inputs, outputs)

    # Quantize model weights to 8 bit and prune the model if both flags are on
    if use_qat and use_pruning:

        # Set up quantization annoations for the pruned model
        quant_aware_annotate_model = tfmot.quantization.keras.quantize_annotate_model(
            pruned_model_unstructured_for_export
        )

        # Quantize the trained model using the training awareness and preserve
        # the pruning scheme while quantizing.
        model = tfmot.quantization.keras.quantize_apply(
            quant_aware_annotate_model,
            tfmot.experimental.combine.Default8BitPrunePreserveQuantizeScheme(),
        )
    
    # Convert the model to a quantization aware model if the quantization flag is turned on
    elif use_qat:
        model = tfmot.quantization.keras.quantize_model(model)

    # Prune the trained model by removing weights that are zero or mostly zero
    elif use_pruning:
        if use_pruning_struct:
            # Set pruning parameters to structured pruning with constant sparsity
            pruning_params = {
                "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
                    0.95, begin_step=1, end_step=-1, frequency=1
                ),
                "block_size": (1, 1),
            }
        else:
             # Set pruning parameters to unstructured pruning with dynamic sparsity
            if use_dynamic_sparsity: 
                pruning_params = {
                    "pruning_schedule": tfmot.sparsity.keras.PolynomialDecay(
                        initial_sparsity=0.30,
                        final_sparsity=0.90,
                        begin_step=10,
                        end_step=400,
                        frequency=10,
                    )
                }
            # Prune with constant sparsity if dynamic sparsity is not enabled
            else:
                pruning_params = {
                    "pruning_schedule": tfmot.sparsity.keras.ConstantSparsity(
                        0.7, begin_step=10, frequency=10
                    ),
                }
        
        # Prune the model based on the provided sparsity parameters
        model = tfmot.sparsity.keras.prune_low_magnitude(model, **pruning_params)

    # Return the model after all architecture, pruning, and quantization techniques
    # have been applied 
    return model

"""Save the provided pruned model into H5 format for later use. """
def save_pruned_model(pruned_model, pruned_keras_file):
    # Strip the model of it's pruning annotations to save space and save the 
    # model file in H5 format to disk
    pruned_model_for_export = tfmot.sparsity.keras.strip_pruning(pruned_model)
    tf.keras.models.save_model(
        pruned_model_for_export, pruned_keras_file, include_optimizer=False
    )
    print("Saved pruned Keras model to:", os.path.abspath(pruned_keras_file))
    return pruned_model_for_export