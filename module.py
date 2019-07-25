import tensorflow as tf

def hardSigmoid(x):
    return tf.nn.relu6(x + 3) / 6

def hardSwish(x):
    return x * hardSigmoid(x)