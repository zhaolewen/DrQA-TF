import tensorflow as tf

def lstm_cell(hidden, drop_keep):
    basic = tf.nn.rnn_cell.LSTMCell(num_units=hidden, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell=basic, output_keep_prob=drop_keep)
    return cell

class StackedBRNN():
    def __init__(self, input_data, hidden_size, num_layers,dropout_rate=0.7):

        outputs = []
        for k in range(num_layers):
            with tf.variable_scope("BiLSTM_"+str(k)):
                with tf.variable_scope("forward"):
                    fw_cell = lstm_cell(hidden_size, dropout_rate)
                    # print(fw_cell.state_size)

                with tf.variable_scope("backward"):
                    bw_cell = lstm_cell(hidden_size, dropout_rate)
                    # print(bw_cell.state_size)

                with tf.name_scope("doc_length"):
                    words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(input_data), reduction_indices=2))
                    self.length = tf.cast(tf.reduce_sum(words_used_in_sent, reduction_indices=1), tf.int32)
                output, _ = tf.nn.bidirectional_dynamic_rnn(fw_cell, bw_cell, input_data, dtype=tf.float32, sequence_length=self.length)
                #print(output)

            outputs += [output[0],output[1]]

        self.output = tf.concat(outputs, axis=2)


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
        with tf.variable_scope('SeqAttnMatch'):

            W = tf.Variable(tf.random_normal(shape=[input_size, input_size], dtype=tf.float32))
            b = tf.Variable(tf.random_normal(([None, input_size]), dtype=tf.float32))

        # Project vectors
        with tf.name_scope("proj_x"):
            x_re = tf.reshape(x, [-1, input_size])
            x_proj = tf.nn.relu(tf.add(tf.matmul(x_re, W),b))
            x_proj = tf.reshape(x_proj, [-1, x.get_shape().as_list()[1], input_size])

        with tf.name_scope("proj_y"):
            y_re = tf.reshape(y, [-1, input_size])
            y_proj = tf.nn.relu(tf.add(tf.matmul(y_re, W),b))
            y_proj = tf.reshape(y_proj, [-1, y.get_shape().as_list()[1], input_size])

        # Compute scores
        scores = tf.matmul(x_proj, y_proj, transpose_b=True)

        # Normalize with softmax
        with tf.name_scope("softmax"):
            alpha_flat = tf.reshape(scores,[-1, y.get_shape().as_list()[1]])

            alpha_flat = tf.exp(alpha_flat)
            z = tf.cast(tf.logical_not(y_mask), tf.float32)
            #z = tf.tile(z, [x.get_shape().as_list()[1],1])
            alpha_flat = tf.multiply(alpha_flat, z)

            alpha_soft = tf.reduce_sum(alpha_flat, axis=1)
            #alpha_soft = tf.clip_by_value(alpha_soft, 1e-7,1e10)
            alpha_soft = tf.expand_dims(alpha_soft, dim=1)
            alpha_soft = tf.tile(alpha_soft, [1,y.get_shape().as_list()[1]])
            alpha_flat = tf.div(alpha_flat, alpha_soft)
            #alpha_flat = tf.clip_by_value(alpha_flat, 1e-7, 1e10)

        #alpha_flat = tf.nn.softmax()
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
            b = tf.Variable(tf.random_normal(([x_size]), dtype=tf.float32))
            Wy = tf.add(tf.matmul(y, W),b)

            xWy = tf.matmul(x,tf.expand_dims(Wy, 2))
            xWy = tf.squeeze(xWy, 2, name="alpha")
            z = tf.cast(tf.logical_not(x_mask), dtype=tf.float32)
            self.alpha = tf.multiply(xWy, z)

class LinearSeqAttn():
    """Self attention over a sequence:
    * o_i = softmax(Wx_i) for x_i in X.
    """

    def __init__(self,x, x_mask):
        """
           x = batch * len * hdim
           x_mask = batch * len
        """
        x_size = x.get_shape().as_list()
        with tf.variable_scope("LinearSaqAttn"):
            W = tf.Variable(tf.truncated_normal([x_size[2],1]))

        with tf.name_scope("Attention"):
            x_flat = tf.reshape(x, [-1, x_size[2]])
            scores = tf.reshape(tf.matmul(x_flat, W),[-1,x_size[1]])
            x_mask = tf.cast(tf.logical_not(x_mask), tf.float32)

            scores = tf.multiply(tf.exp(scores),x_mask)
            x_sum = tf.expand_dims(tf.reduce_sum(scores, axis=1), axis=1)
            #x_sum = tf.tile(x_sum, [1,x_size[1]])

            scores = tf.expand_dims(tf.divide(scores, x_sum), axis=1)
            self.weighted = tf.squeeze(tf.matmul(scores, x), axis=1)