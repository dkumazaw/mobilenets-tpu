import collections

BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'hidden_filters', 'output_filters',
    'id_skip', 'strides', 'se_ratio', 'nonlinearity'
])

# Override the default.
# defaults will be a public argument for namedtuple in Python 3.7
# https://docs.python.org/3/library/collections.html#collections.namedtuple
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)

GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate', 'data_format',
    'num_classes', 'depth_multiplier', 'depth_divisor', 'min_depth',
])

GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)