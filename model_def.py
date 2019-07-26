import tensorflow as tf

from defs import BlockArgs, GlobalParams
from module import hardSigmoid, hardSwish
from ops import *

class V3Block(object):
    """A class of MobileNetV3  Inverted Residual Bottleneck."""

    def __init__(self, block_args, global_params):
        """
        Args:
            block_args: BlockArgs, arguments to create a V3 block.
            global_params: GlobalParams, a set of global parameters.
        """
        self._block_args = block_args
        self._batch_norm_momentum = global_params.batch_norm_momentum
        self._batch_norm_epsilon = global_params.batch_norm_epsilon
        if global_params.data_format == 'channels_first':
            self._channel_axis = 1
            self._spatial_dims = [2, 3]
        else:
            self._channel_axis = -1
            self._spatial_dims = [1, 2]
        self.has_se = (self._block_args.se_ratio is not None) and (
            self._block_args.se_ratio > 0) and (self._block_args.se_ratio <= 1)

        self.nonlinearity = hardSwish if block_args.nonlinearity == 'HS' else tf.nn.relu6

        self.endpoints = None

        # Builds the block accordings to arguments.
        self._build()

    def _build(self):
        """Builds the V3 block according to the arguments."""
        filters = self._block_args.hidden_filters
        # Expansion phase:
        self._expand_conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn0 = tf.layers.BatchNormalization(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
            fused=True)

        kernel_size = self._block_args.kernel_size
        # Depth-wise convolution phase:
        self._depthwise_conv = tf.keras.layers.DepthwiseConv2D(
            [kernel_size, kernel_size],
            strides=self._block_args.strides,
            depthwise_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn1 = tf.layers.BatchNormalization(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
            fused=True)

        if self.has_se:
            num_reduced_filters = max(
                1, int(self._block_args.input_filters * self._block_args.se_ratio))
            # Squeeze and Excitation layer.
            self._se_reduce = tf.keras.layers.Conv2D(
                num_reduced_filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=True)
            self._se_expand = tf.keras.layers.Conv2D(
                filters,
                kernel_size=[1, 1],
                strides=[1, 1],
                kernel_initializer=conv_kernel_initializer,
                padding='same',
                use_bias=True)

        # Output phase:
        filters = self._block_args.output_filters
        self._project_conv = tf.keras.layers.Conv2D(
            filters,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn2 = tf.layers.BatchNormalization(
            axis=self._channel_axis,
            momentum=self._batch_norm_momentum,
            epsilon=self._batch_norm_epsilon,
            fused=True)

    def _call_se(self, input_tensor):
        """Call Squeeze and Excitation layer.
        Args:
          input_tensor: Tensor, a single input tensor for Squeeze/Excitation layer.
        Returns:
          A output tensor, which should have the same shape as input.
        """
        se_tensor = tf.reduce_mean(
            input_tensor, self._spatial_dims, keepdims=True)
        se_tensor = self._se_expand(tf.nn.relu6(self._se_reduce(se_tensor)))
        tf.logging.info('Built Squeeze and Excitation with tensor shape: %s' %
                        (se_tensor.shape))
        return hardSigmoid(se_tensor) * input_tensor

    def call(self, inputs, training=True):
        """Implementation of Block call().
        Args:
          inputs: the inputs tensor.
          training: boolean, whether the model is constructed for training.
        Returns:
          A output tensor.
        """
        tf.logging.info('Block input: %s shape: %s' %
                        (inputs.name, inputs.shape))
        x = self.nonlinearity(
            self._bn0(self._expand_conv(inputs), training=training))
        
        tf.logging.info('Expand: %s shape: %s' % (x.name, x.shape))

        x = self.nonlinearity(self._bn1(self._depthwise_conv(x), training=training))
        tf.logging.info('DWConv: %s shape: %s' % (x.name, x.shape))

        if self.has_se:
            with tf.variable_scope('se'):
                x = self._call_se(x)

        self.endpoints = {'expansion_output': x}

        x = self._bn2(self._project_conv(x), training=training)
        
        # Identity op
        if self._block_args.id_skip:
            if (self._block_args.strides == 1) and self._block_args.input_filters == self._block_args.output_filters:
                x = tf.add(x, inputs)
        tf.logging.info('Project: %s shape: %s' % (x.name, x.shape))
        return x


class MobileNetV3Small(tf.keras.Model):
    """Implements tf.keras.Model for MobileNetV3Small."""

    def __init__(self, global_params=None):
        """
        Args:
            lobal_params: GlobalParams, a set of global parameters.
        Raises:
            ValueError: when blocks_args is not specified as a list.
        """
        super().__init__()
        self._blocks_args = \
            [BlockArgs(3, 1, 16,  16, 16, True, 2, 0.25, 'RE'),
             BlockArgs(3, 1, 16,  72, 24, True, 2, None, 'RE'),
             BlockArgs(3, 1, 24,  88, 24, True, 1, None, 'RE'),
             BlockArgs(5, 1, 24,  96, 40, True, 2, 0.25, 'HS'),
             BlockArgs(5, 1, 40, 240, 40, True, 1, 0.25, 'HS'),
             BlockArgs(5, 1, 40, 240, 40, True, 1, 0.25, 'HS'),
             BlockArgs(5, 1, 40, 120, 48, True, 1, 0.25, 'HS'),
             BlockArgs(5, 1, 48, 144, 48, True, 1, 0.25, 'HS'),
             BlockArgs(5, 1, 48, 288, 96, True, 2, 0.25, 'HS'),
             BlockArgs(5, 1, 96, 576, 96, True, 1, 0.25, 'HS'),
             BlockArgs(5, 1, 96, 576, 96, True, 1, 0.25, 'HS')
            ]

        self._global_params = global_params
        self.endpoints = None
        self._build()

    def _build(self):
        """Builds a model."""
        self._blocks = []
        # Builds blocks.
        for block_args in self._blocks_args:
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters,
                                            self._global_params),
                output_filters=round_filters(block_args.output_filters,
                                             self._global_params))

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(V3Block(block_args, self._global_params))
            if block_args.num_repeat > 1:
                # pylint: disable=protected-access
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1])
                # pylint: enable=protected-access
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(V3Block(block_args, self._global_params))

        batch_norm_momentum = self._global_params.batch_norm_momentum
        batch_norm_epsilon = self._global_params.batch_norm_epsilon
        if self._global_params.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        # Stem part.
        self._conv_stem = tf.keras.layers.Conv2D(
            filters=round_filters(16, self._global_params),
            kernel_size=[3, 3],
            strides=[2, 2],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn0 = tf.layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon,
            fused=True)

        # Head part.
        self._conv_expand = tf.keras.layers.Conv2D(
            filters=576,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn1 = tf.layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon,
            fused=True)
        self._avg_pooling = tf.keras.layers.AveragePooling2D(
            pool_size=[7, 7],
            data_format=self._global_params.data_format)
        self._conv_head = tf.keras.layers.Conv2D(
            filters=1280,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._final = tf.keras.layers.Conv2D(
            filters=self._global_params.num_classes,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)

        if self._global_params.dropout_rate > 0:
            self._dropout = tf.keras.layers.Dropout(
                self._global_params.dropout_rate)
        else:
            self._dropout = None

    def call(self, inputs, training=True):
        """Implementation of MobileNetV3 call().
        Args:
            inputs: input tensors.
            training: boolean, whether the model is constructed for training.
        Returns:
            output tensors.
        """
        outputs = None
        self.endpoints = {}
        # Calls Stem layers
        with tf.variable_scope('v3_stem'):
            outputs = hardSwish(
                self._bn0(self._conv_stem(inputs), training=training))
        tf.logging.info('Built stem layers with output shape: %s' %
                        outputs.shape)
        self.endpoints['stem'] = outputs
        # Calls blocks.
        for idx, block in enumerate(self._blocks):
            with tf.variable_scope('v3_blocks_%s' % idx):
                outputs = block.call(outputs, training=training)
                self.endpoints['block_%s' % idx] = outputs
                if block.endpoints:
                    for k, v in block.endpoints.items():
                        self.endpoints['block_%s/%s' % (idx, k)] = v
        # Calls final layers and returns logits.
        with tf.variable_scope('v3_head'):
            outputs = hardSwish(
                self._bn1(self._conv_expand(outputs), training=training))
            outputs = self._avg_pooling(outputs)
            outputs = hardSwish(self._conv_head(outputs))
            if self._dropout:
                outputs = self._dropout(outputs, training=training)
            outputs = tf.reshape(self._final(outputs), [-1, self._global_params.num_classes])
            self.endpoints['head'] = outputs
        return outputs

class MobileNetV3Large(tf.keras.Model):
    """Implements tf.keras.Model for MobileNetV3Large."""

    def __init__(self, global_params=None):
        """
        Args:
            lobal_params: GlobalParams, a set of global parameters.
        Raises:
            ValueError: when blocks_args is not specified as a list.
        """
        super().__init__()
        self._blocks_args = \
            [BlockArgs(3, 1,  16,  16,  16, True, 1, None, 'RE'),
             BlockArgs(3, 1,  16,  64,  24, True, 2, None, 'RE'),
             BlockArgs(3, 1,  24,  72,  24, True, 1, None, 'RE'),
             BlockArgs(5, 1,  24,  72,  40, True, 2, 0.25, 'RE'),
             BlockArgs(5, 1,  40, 120,  40, True, 1, 0.25, 'RE'),
             BlockArgs(5, 1,  40, 120,  40, True, 1, 0.25, 'RE'),
             BlockArgs(3, 1,  40, 240,  80, True, 2, None, 'HS'),
             BlockArgs(3, 1,  80, 200,  80, True, 1, None, 'HS'),
             BlockArgs(3, 1,  80, 184,  80, True, 1, None, 'HS'),
             BlockArgs(3, 1,  80, 184,  80, True, 1, None, 'HS'),
             BlockArgs(3, 1,  80, 480, 112, True, 1, 0.25, 'HS'),
             BlockArgs(3, 1, 112, 672, 112, True, 1, 0.25, 'HS'),
             BlockArgs(5, 1, 112, 672, 160, True, 2, 0.25, 'HS'),
             BlockArgs(5, 1, 160, 960, 160, True, 1, 0.25, 'HS'),
             BlockArgs(5, 1, 160, 960, 160, True, 1, 0.25, 'HS')
            ]

        self._global_params = global_params
        self.endpoints = None
        self._build()

    def _build(self):
        """Builds a model."""
        self._blocks = []
        # Builds blocks.
        for block_args in self._blocks_args:
            assert block_args.num_repeat > 0
            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters,
                                            self._global_params),
                output_filters=round_filters(block_args.output_filters,
                                             self._global_params))

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(V3Block(block_args, self._global_params))
            if block_args.num_repeat > 1:
                # pylint: disable=protected-access
                block_args = block_args._replace(
                    input_filters=block_args.output_filters, strides=[1, 1])
                # pylint: enable=protected-access
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(V3Block(block_args, self._global_params))

        batch_norm_momentum = self._global_params.batch_norm_momentum
        batch_norm_epsilon = self._global_params.batch_norm_epsilon
        if self._global_params.data_format == 'channels_first':
            channel_axis = 1
        else:
            channel_axis = -1

        # Stem part.
        self._conv_stem = tf.keras.layers.Conv2D(
            filters=round_filters(16, self._global_params),
            kernel_size=[3, 3],
            strides=[2, 2],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn0 = tf.layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon,
            fused=True)

        # Head part.
        self._conv_expand = tf.keras.layers.Conv2D(
            filters=960,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._bn1 = tf.layers.BatchNormalization(
            axis=channel_axis,
            momentum=batch_norm_momentum,
            epsilon=batch_norm_epsilon,
            fused=True)
        self._avg_pooling = tf.keras.layers.AveragePooling2D(
            pool_size=[7, 7],
            data_format=self._global_params.data_format)
        self._conv_head = tf.keras.layers.Conv2D(
            filters=1280,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)
        self._final = tf.keras.layers.Conv2D(
            filters=self._global_params.num_classes,
            kernel_size=[1, 1],
            strides=[1, 1],
            kernel_initializer=conv_kernel_initializer,
            padding='same',
            use_bias=False)

        if self._global_params.dropout_rate > 0:
            self._dropout = tf.keras.layers.Dropout(
                self._global_params.dropout_rate)
        else:
            self._dropout = None

    def call(self, inputs, training=True):
        """Implementation of MobileNetV3 call().
        Args:
            inputs: input tensors.
            training: boolean, whether the model is constructed for training.
        Returns:
            output tensors.
        """
        outputs = None
        self.endpoints = {}
        # Calls Stem layers
        with tf.variable_scope('v3_stem'):
            outputs = hardSwish(
                self._bn0(self._conv_stem(inputs), training=training))
        tf.logging.info('Built stem layers with output shape: %s' %
                        outputs.shape)
        self.endpoints['stem'] = outputs
        # Calls blocks.
        for idx, block in enumerate(self._blocks):
            with tf.variable_scope('v3_blocks_%s' % idx):
                outputs = block.call(outputs, training=training)
                self.endpoints['block_%s' % idx] = outputs
                if block.endpoints:
                    for k, v in block.endpoints.items():
                        self.endpoints['block_%s/%s' % (idx, k)] = v
        # Calls final layers and returns logits.
        with tf.variable_scope('v3_head'):
            outputs = hardSwish(
                self._bn1(self._conv_expand(outputs), training=training))
            outputs = self._avg_pooling(outputs)
            outputs = hardSwish(self._conv_head(outputs))
            if self._dropout:
                outputs = self._dropout(outputs, training=training)
            outputs = tf.reshape(self._final(outputs), [-1, self._global_params.num_classes])
            self.endpoints['head'] = outputs
        return outputs