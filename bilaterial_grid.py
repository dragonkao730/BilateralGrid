import tensorflow as tf


def get_tensor_shape(x):
    a = x.get_shape().as_list()
    b = [tf.shape(x)[i] for i in range(len(a))]
    r = []
    for aa, bb in zip(a, b):
        r.append(aa if type(aa) is int else bb)
    return r
