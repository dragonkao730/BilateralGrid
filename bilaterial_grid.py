import tensorflow as tf

def get_tensor_shape(x):
    a = x.get_shape().as_list()
    b = [tf.shape(x)[i] for i in range(len(a))]
    r = []
    return [aa if type(aa) is int else bb for aa, bb in zip(a, b)]

# [b, n, c]
def sample_1d(
    img,   # [b, h, c]
    y_idx, # [b, n], 0 <= pos < h, dtpye=int32
):
    b, h, c = get_tensor_shape(img)
    b, n    = get_tensor_shape(y_idx)
    
    b_idx = tf.range(b, dtype=tf.int32) # [b]
    b_idx = tf.expand_dims(b_idx, -1)   # [b, 1]
    b_idx = tf.tile(b_idx, [1, n])      # [b, n]
    
    y_idx = tf.clip_by_value(y_idx, 0, h - 1) # [b, n]
    a_idx = tf.stack([b_idx, y_idx], axis=-1) # [b, n, 2]
    
    return tf.gather_nd(img, a_idx)

# [b, n, c]
def interp_1d(
    img, # [b, h, c]
    y,   # [b, n], 0 <= pos < h, dtype=float32
):
    b, h, c = get_tensor_shape(img)
    b, n    = get_tensor_shape(y)
    
    y_0 = tf.floor(y) # [b, n]
    y_1 = y_0 + 1    
    
    _sample_func = lambda y_x: sample_1d(
        img,
        tf.cast(y_x, tf.int32)
    )
    y_0_val = _sample_func(y_0) # [b, n, c]
    y_1_val = _sample_func(y_1)
    
    w_0 = y_1 - y # [b, n]
    w_1 = y - y_0
    
    w_0 = tf.expand_dims(w_0, -1) # [b, n, 1]
    w_1 = tf.expand_dims(w_1, -1)
    
    return w_0*y_0_val + w_1*y_1_val

# [b, h, w, 3]
def apply_bg(
    bg,     # [b, ?, ?, d*3*4]
    guide,  # [b, h, w], 0 <= guide <= 1
    in_img, # [b, h, w, 3]
):
    b, _, _, d34, = get_tensor_shape(bg)
    b, h, w,      = get_tensor_shape(guide)
    b, h, w, _,   = get_tensor_shape(in_img)
    
    d = d34//3//4
    
    bg = tf.image.resize_images(bg, [h, w]) # [b, h, w, d*3*4]
    
    coef = interp_1d(
        tf.reshape(bg, [b*h*w, d, 3*4]),       # [b*h*w, d, 3*4]
        (d - 1)*tf.reshape(guide, [b*h*w, 1]), # [b*h*w, 1]
    ) # [b*h*w, 1, 3*4]
    coef = tf.reshape(coef, [b, h, w, 3, 4]) # [b, h, w, 3, 4]
    
    in_img = tf.reshape(in_img, [b, h, w, 3, 1]) # [b, h, w, 3, 1]
    in_img = tf.pad(
        in_img,
        [[0, 0], [0, 0], [0, 0], [0, 1], [0, 0]],
        mode='CONSTANT',
        constant_values=1,
    ) # [b, h, w, 4, 1]
    
    out_img = tf.matmul(coef, in_img)           # [b, h, w, 3, 1]
    out_img = tf.reshape(out_img, [b, h, w, 3]) # [b, h, w, 3]
    
    return out_img
    
    
