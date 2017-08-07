import tensorflow as tf
import numpy as np

tf.reset_default_graph()

# Create input data
X = np.random.randn(2, 10, 8)

# The second example is of length 6
X[1,6,:] = 0
X_lengths = [10, 6]

def lstm_cell():
    basic = tf.nn.rnn_cell.LSTMCell(num_units=64, state_is_tuple=True)
    cell = tf.nn.rnn_cell.DropoutWrapper(cell=basic, output_keep_prob=0.5)
    return cell

stacked_lstm = tf.contrib.rnn.MultiRNNCell([lstm_cell() for _ in range(3)])

outputs, last_states = tf.nn.dynamic_rnn(
    cell=stacked_lstm,
    dtype=tf.float64,
    sequence_length=X_lengths,
    inputs=X)

result = tf.contrib.learn.run_n(
    {"outputs": outputs, "last_states": last_states},
    n=1,
    feed_dict=None)


print(result[0]["outputs"].shape)
print(result[0]["outputs"])
assert result[0]["outputs"].shape == (2, 10, 64)

# Outputs for the second example past past length 6 should be 0
assert (result[0]["outputs"][1,7,:] == np.zeros(stacked_lstm.output_size)).all()

print(result[0]["last_states"][0].h.shape)
print(result[0]["last_states"][0].h)