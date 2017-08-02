import tensorflow as tf

class StackedBRNN():
    def __init__(self, input_data, hidden_size, num_layers,dropout_rate=0.1):

        with tf.variable_scope("forward"):
            fw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)

            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=dropout_rate)
            fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell] * num_layers, state_is_tuple=True)
            print(fw_cell.state_size)

        with tf.variable_scope("backward"):
            bw_cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True)

            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=dropout_rate)
            bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell] * num_layers, state_is_tuple=True)
            print(bw_cell.state_size)

        words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(input_data), reduction_indices=2))
        self.length = tf.cast(tf.reduce_sum(words_used_in_sent, reduction_indices=1), tf.int32)
        output, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input_data, dtype=tf.float32, sequence_length=self.length)
        print(output)

        self.output = output


class SeqAttnMatch():
    """Given sequences X and Y, match sequence Y to each element in X.
    * o_i = sum(alpha_j * y_j) for i in X
    * alpha_j = softmax(y_j * x_i)
    """
    def __init__(self, input_size, x, y, y_mask):
        """
        Input shapes:
            x = batch * len1 * h
            y = batch * len2 * h
            y_mask = batch * len2
        Output shapes:
            matched_seq = batch * len1 * h
        """
        with tf.variable_scope('BilinearSeqAttention'):

            W = tf.Variable(tf.random_normal(shape=[input_size, input_size], dtype=tf.float32))
            # b = tf.Variable(tf.random_normal(([None, input_size]), dtype=tf.float32))

        # Project vectors
        x_re = tf.reshape(x, [-1, input_size])
        x_proj = tf.nn.relu(tf.matmul(x_re, W))
        x_proj = tf.reshape(x_proj, [-1, x.get_shape().as_list()[1], input_size])

        y_re = tf.reshape(y, [-1, input_size])
        y_proj = tf.nn.relu(tf.matmul(y_re, W))
        y_proj = tf.reshape(y_proj, [-1, y.get_shape().as_list()[1], input_size])

        # Compute scores
        scores = tf.matmul(x_proj, y_proj, transpose_b=True)

        # Normalize with softmax
        alpha_flat = tf.nn.softmax(tf.reshape(scores,[-1, y.get_shape().as_list()[1]]))
        alpha = tf.reshape(alpha_flat,[-1, x.get_shape().as_list()[1], y.get_shape().as_list()[1]])

        # Take weighted average
        self.matched_seq = tf.matmul(alpha, y)


class BilinearSeqAttn():
    """A bilinear attention layer over a sequence X w.r.t y:
    * o_i = softmax(x_i'Wy) for x_i in X.

    Optionally don't normalize output weights.
    """
    def __init__(self, x_size, y_size, x, y, x_mask):
        """
            x = batch * len * h1
            y = batch * h2
            x_mask = batch * len
        """
        with tf.variable_scope('BilinearSeqAttention'):
            W = tf.Variable(tf.truncated_normal([y_size,x_size], dtype=tf.float32))
            #b = tf.Variable(tf.random_normal(([x_size]), dtype=tf.float32))
            # Wy = tf.matmul(y, W)
            #Wy = W * y + b

            Wy = tf.matmul(y, W)

            xWy = tf.matmul(x,tf.expand_dims(Wy, 2))
            self.alpha = tf.squeeze(xWy, 2, name="alpha")

            #self.alpha = tf.nn.softmax(xWy)


def uniform_weights(x, x_mask):
    """Return uniform weights over non-masked input."""
    alpha = tf.ones(x.get_shape().as_list(), dtype=tf.float32)
    alpha = tf.multiply(alpha, tf.equal(x_mask, 0.0))

    sums = tf.reduce_sum(alpha, 1)
    sums = tf.tile(sums, alpha.get_shape().as_list()[1])
    alpha = tf.div(alpha, sums)
    return alpha


def weighted_avg(x, weights):
    """x = batch * len * d
    weights = batch * len
    """
    wx = tf.matmul(tf.expand_dims(weights, 1), x)
    return tf.squeeze(wx, 1)
