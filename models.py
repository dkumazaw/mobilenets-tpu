
# This project incorporates material from the project listed above, and it
# is accessible under their original license terms (Apache License 2.0)
# ==============================================================================
"""Creates the ConvNet"""

import re
import tensorflow as tf

import numpy as np

from defs import GlobalParams
import model_def

def build_model(images, model_name, training, override_params=None):
    """A helper functiion to creates a ConvNet model and returns predicted logits.
    Args:
        images: input images tensor.
        model_name: string, the model name (either MobileNetV3Large or MobileNetV3Small).
        training: boolean, whether the model is constructed for training.
        override_params: A dictionary of params for overriding. Fields must exist in
            EvalGlobalParams.
    Returns:
        logits: the logits tensor of classes.
        endpoints: the endpoints for each layer.
    Raises:
        When model_name specified an undefined model, raises NotImplementedError.
        When override_params has invalid fields, raises ValueError.
    """
    assert isinstance(images, tf.Tensor)

    global_params = EvalGlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=0.2,
        data_format='channels_last',
        num_classes=1000,
        depth_multiplier=depth_multiplier,
        depth_divisor=8,
        min_depth=None)

    if override_params:
        # ValueError will be raised here if override_params has fields not included
        # in global_params.
        global_params = global_params._replace(**override_params)

    if model_name.lower() != 'mobilenetv3small':
        raise NotImplementedError
        
    with tf.variable_scope(model_name):
        model = model_def.MobileNetV3Small(global_params)
        logits = model(images, training=training)

    logits = tf.identity(logits, 'logits')
    return logits, model.endpoints
