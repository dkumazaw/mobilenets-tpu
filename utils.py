import json
import logging

from google.cloud import storage
import numpy as np
import tensorflow as tf

from dataset import imagenet

logger = logging.getLogger(__name__)

def build_learning_rate(initial_lr,
                        global_step,
                        steps_per_epoch=None,
                        lr_decay_type='exponential',
                        decay_factor=0.97,
                        decay_epochs=2.4,
                        total_steps=None,
                        warmup_epochs=5):
    """Build learning rate."""
    if lr_decay_type == 'exponential':
        assert steps_per_epoch is not None
        decay_steps = steps_per_epoch * decay_epochs
        lr = tf.train.exponential_decay(
            initial_lr, global_step, decay_steps, decay_factor, staircase=True)
    elif lr_decay_type == 'cosine':
        assert total_steps is not None
        lr = 0.5 * initial_lr * (
            1 + tf.cos(np.pi * tf.cast(global_step, tf.float32) / total_steps))
    elif lr_decay_type == 'constant':
        lr = initial_lr
    else:
        assert False, 'Unknown lr_decay_type : %s' % lr_decay_type

    if warmup_epochs:
        tf.logging.info('Learning rate warmup_epochs: %d' % warmup_epochs)
        warmup_steps = int(warmup_epochs * steps_per_epoch)
        warmup_lr = (
            initial_lr * tf.cast(global_step, tf.float32) / tf.cast(
                warmup_steps, tf.float32))
        lr = tf.cond(global_step < warmup_steps, lambda: warmup_lr, lambda: lr)

    return lr


def build_dropout_rate(global_step, warmup_steps=2502):
    tf.logging.info('Dropout rate warmup steps: %d' % warmup_steps)
    warmup_dropout_rate = tf.cast(0.6, tf.float32)
    final_dropout_rate = tf.cast(1e2, tf.float32)
    dropout_rate = tf.cond(global_step < warmup_steps, lambda: warmup_dropout_rate,
                           lambda: final_dropout_rate)
    return dropout_rate

def build_optimizer(learning_rate,
                    optimizer_name='rmsprop',
                    decay=0.9,
                    epsilon=0.001,
                    momentum=0.9):
    """Build optimizer."""
    if optimizer_name == 'sgd':
        tf.logging.info('Using SGD optimizer')
        optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=learning_rate)
    elif optimizer_name == 'momentum':
        tf.logging.info('Using Momentum optimizer')
        optimizer = tf.train.MomentumOptimizer(
            learning_rate=learning_rate, momentum=momentum)
    elif optimizer_name == 'rmsprop':
        tf.logging.info('Using RMSProp optimizer')
        optimizer = tf.train.RMSPropOptimizer(learning_rate, decay, momentum,
                                              epsilon)
    else:
        tf.logging.fatal('Unknown optimizer:', optimizer_name)

    return optimizer


# NOTE: Kept for potential bigtable support
def verify_non_empty_string(value, field_name):
    """Ensures that a given proposed field value is a non-empty string.
    Args:
        value:  proposed value for the field.
        field_name:  string name of the field, e.g. `project`.
    Returns:
        The given value, provided that it passed the checks.
    Raises:
        ValueError:  the value is not a string, or is a blank string.
    """
    if not isinstance(value, str):
        raise ValueError(
            'Bigtable parameter "%s" must be a string.' % field_name)
    if not value:
        raise ValueError(
            'Bigtable parameter "%s" must be non-empty.' % field_name)
    return value


def select_tables_from_flags(FLAGS):
    """Construct training and evaluation Bigtable selections from flags.
    Args:
        FLAGS: An abseil flags instance
    Returns:
        [training_selection, evaluation_selection]
    """
    project = verify_non_empty_string(
        FLAGS.bigtable_project or FLAGS.gcp_project,
        'project')
    instance = verify_non_empty_string(FLAGS.bigtable_instance, 'instance')
    table = verify_non_empty_string(FLAGS.bigtable_table, 'table')
    train_prefix = verify_non_empty_string(FLAGS.bigtable_train_prefix,
                                           'train_prefix')
    eval_prefix = verify_non_empty_string(FLAGS.bigtable_eval_prefix,
                                          'eval_prefix')
    column_family = verify_non_empty_string(FLAGS.bigtable_column_family,
                                            'column_family')
    column_qualifier = verify_non_empty_string(FLAGS.bigtable_column_qualifier,
                                               'column_qualifier')
    return [
        imagenet.BigtableSelection(
            project=project,
            instance=instance,
            table=table,
            prefix=p,
            column_family=column_family,
            column_qualifier=column_qualifier)
        for p in (train_prefix, eval_prefix)
    ]


def prepare_input_pipeline(FLAGS):
    """Prepares dataset input pipelines. 
    Returns:
        Instances of ImageNetInput or ImageNetBigtableInput for train and eval
    """
    # NOTE: Kept for potential bigtable support
    # Input pipelines are slightly different (with regards to shuffling and
    # preprocessing) between training and evaluation.
    if FLAGS.bigtable_instance:
        tf.logging.info('Using Bigtable dataset, table %s',
                        FLAGS.bigtable_table)
        select_train, select_eval = utils.select_tables_from_flags()
        imagenet_train, imagenet_eval = [imagenet.ImageNetBigtableInput(
            is_training=is_training,
            use_bfloat16=False,
            transpose_input=FLAGS.transpose_input,
            selection=selection) for (is_training, selection) in
            [(True, select_train),
             (False, select_eval)]]

    # Use an Imagenet bucket
    else:
        tf.logging.info('Using dataset: %s', FLAGS.data_dir)
        imagenet_train, imagenet_eval = [
            imagenet.ImageNetInput(
                is_training=is_training,
                data_dir=FLAGS.data_dir,
                transpose_input=FLAGS.transpose_input,
                cache=FLAGS.use_cache and is_training,
                image_size=FLAGS.input_image_size,
                num_parallel_calls=FLAGS.num_parallel_calls,
                use_bfloat16=False) for is_training in [True, False]
        ]

    return imagenet_train, imagenet_eval


def get_override_params_dict(FLAGS):
    """Parses the input flags and generates a dict of parameters to be overriden
    Args:
        FLAGS:
    Returns:
        dict: A dictionary of overriden parameters
    """
    override_params = {}
    if FLAGS.batch_norm_momentum:
        override_params['batch_norm_momentum'] = FLAGS.batch_norm_momentum
    if FLAGS.batch_norm_epsilon:
        override_params['batch_norm_epsilon'] = FLAGS.batch_norm_epsilon
    if FLAGS.dropout_rate:
        override_params['dropout_rate'] = FLAGS.dropout_rate
    if FLAGS.data_format:
        override_params['data_format'] = FLAGS.data_format
    if FLAGS.num_label_classes:
        override_params['num_classes'] = FLAGS.num_label_classes
    if FLAGS.depth_multiplier:
        override_params['depth_multiplier'] = FLAGS.depth_multiplier
    # TODO: Fix this
    #if FLAGS.kernel:
    #    override_params['kernel'] = FLAGS.kernel
    #if FLAGS.expratio:
    #    override_params['expratio'] = FLAGS.expratio
    if FLAGS.depth_divisor:
        override_params['depth_divisor'] = FLAGS.depth_divisor
    if FLAGS.min_depth:
        override_params['min_depth'] = FLAGS.min_depth

    return override_params

def _parse_gcs_path(path: str):
    """Parses the provided GCS path into bucket name and blob name
    Args:
        path (str): Path to the GCS object
    
    Returns:
        bucket_name, blob_name: Strings denoting the bucket name and blob name
    """

    header, rest = path.strip().split('//')
    if header != 'gs:':
        raise ValueError('Invalid GCS object path: %s', header)

    bucket_name, blob_name = rest.split('/', 1)
    return bucket_name, blob_name