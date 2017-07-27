import numpy as np
import logging
import tensorflow as tf

from .rnn_reader import RnnDocReader

logger = logging.getLogger(__name__)

class DocReaderModel():
    """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
    """
    def __init__(self, opt, embedding=None):
        self.opt = opt
        self.global_step = tf.Variable(0, name="global_step", trainable=False)

        len_d = opt['context_len']
        feat_n = opt['feature_len']
        len_q = opt['question_len']
        self.doc_words = tf.placeholder(tf.int32, [None,len_d], name="document_words")
        self.word_feat = tf.placeholder(tf.float32, [None, len_d, feat_n], name="word_features")
        self.pos = tf.placeholder(tf.int32, [None, len_d],name="pos_features")
        self.ner = tf.placeholder(tf.int32, [None, len_d], name="ner_features")
        self.doc_mask = tf.placeholder(tf.int32, [None, len_d],name="doc_mask")
        self.q_words = tf.placeholder(tf.int32, [None, len_q],name="question_words")
        self.q_mask = tf.placeholder(tf.int32, [None, len_q],name="question_mask")

        self.target_s = tf.placeholder(tf.int32,[None,], name="target_start")
        self.target_e = tf.placeholder(tf.int32,[None,], name="target_end")

        # Building network.
        #  x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, opt
        self.network = RnnDocReader(self.doc_words, self.word_feat,self.pos, self.ner, self.doc_mask, self.q_words, self.q_mask ,opt, embedding=embedding)

        # Run forward
        self.score_s, self.score_e = self.network.start_scores, self.network.end_scores

        t_start = tf.one_hot(self.target_s, depth=len_d, name="target_start")
        t_end = tf.one_hot(self.target_e, depth=len_d, name="target_end")

        # Compute loss and accuracies
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t_start, logits=self.score_s)) \
               + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=t_end, logits=self.score_e))

        self.optimizer = tf.train.AdamOptimizer(opt['learning_rate'])

        gvs = self.optimizer.compute_gradients(loss)
        val = self.opt['grad_clipping']
        capped_gvs = [(tf.clip_by_value(grad, -val, val), var) for grad, var in gvs]

        self.train_op = self.optimizer.apply_gradients(capped_gvs, global_step=self.global_step)

        loss_summary = tf.summary.scalar("loss", loss)

        self.train_summary_op = tf.summary.merge([loss_summary])
        self.test_summary_op = tf.summary.merge([loss_summary])

    def train(self, batch, sess):
        feed_dict = {
            self.doc_words:batch[0], self.word_feat:batch[1], self.pos:batch[2],
            self.ner:batch[3], self.doc_mask:batch[4], self.q_words:batch[5], self.q_mask:batch[6], self.target_s:batch[7], self.target_e:batch[8]
        }

        ops = [self.global_step, self.train_summary_op, self.train_op]

        return sess.run(ops, feed_dict=feed_dict)

    def test(self, batch, sess):
        feed_dict = {
            self.doc_words: batch[0], self.word_feat: batch[1], self.pos: batch[2],
            self.ner: batch[3], self.doc_mask: batch[4], self.q_words: batch[5], self.q_mask: batch[6]
        }

        ops = [self.score_s, self.score_e]

        sc_s, sc_e = sess.run(ops, feed_dict=feed_dict)

        # Get argmax text spans
        text = batch[-2]
        spans = batch[-1]
        predictions = []

        max_len = self.opt['max_len'] or sc_s.size(1)
        for i in range(len(sc_s)):
            scores = np.outer(sc_s[i], sc_e[i])
            scores = scores.triu().tril(max_len - 1)

            s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
            s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
            predictions.append(text[i][s_offset:e_offset])

        return predictions

    def predict(self, sc_start, sc_end, text):
        idx_start = tf.argmax(sc_start, axis=1)
        idx_end = tf.arg_max(sc_end, axis=1)



