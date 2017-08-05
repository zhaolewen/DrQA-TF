import tensorflow as tf
from . import layers

class RnnDocReader():

    def __init__(self,  x1, x1_f, x1_pos, x1_ner, x1_mask, x2, x2_mask, opt, embedding=None):
        """Inputs:
        x1 = document word indices             [batch * len_d]
        x1_f = document word features indices  [batch * len_d * nfeat]
        x1_pos = document POS tags             [batch * len_d]
        x1_ner = document entity tags          [batch * len_d]
        x1_mask = document padding mask        [batch * len_d]
        x2 = question word indices             [batch * len_q]
        x2_mask = question padding mask        [batch * len_q]
        """
        self.opt = opt

        # Word embeddings
        if opt['pretrained_words']:
            assert embedding is not None
            self.embedding = tf.constant(embedding, dtype=tf.float32, name="word_embed")
        else:
            self.embedding = tf.Variable(tf.random_normal((opt['vocab_size'], opt['embedding_dim']), 0.0, 1.0))

        # Embed both document and question
        x1_emb = tf.nn.embedding_lookup(self.embedding, x1)
        x2_emb = tf.nn.embedding_lookup(self.embedding, x2)

        drnn_input_list = [x1_emb, x1_f]
        # Add attention-weighted question representation
        self.pos_embedding = tf.Variable(tf.random_normal((opt['pos_size'], opt['pos_dim']), 0.0, 1.0),name="pos_embed")
        self.ner_embedding = tf.Variable(tf.random_normal((opt['ner_size'], opt['ner_dim']), 0.0, 1.0),name="ner_embed")

        if self.opt['use_qemb']:
            # Projection for attention weighted question
            self.qemb_match = layers.SeqAttnMatch(opt['embedding_dim'], x1_emb, x2_emb, x2_mask)

            x2_weighted_emb = self.qemb_match.matched_seq
            drnn_input_list.append(x2_weighted_emb)

        if self.opt['pos']:
            x1_pos_emb = tf.nn.embedding_lookup(self.pos_embedding, x1_pos)
            drnn_input_list.append(x1_pos_emb)
        if self.opt['ner']:
            x1_ner_emb = tf.nn.embedding_lookup(self.ner_embedding, x1_ner)
            drnn_input_list.append(x1_ner_emb)

        drnn_input = tf.concat(drnn_input_list, axis=2)
        # Input size to RNN: word emb + question emb + manual features
        doc_input_size = opt['embedding_dim'] + opt['num_features']
        if opt['use_qemb']:
            doc_input_size += opt['embedding_dim']
        if opt['pos']:
            doc_input_size += opt['pos_dim']
        if opt['ner']:
            doc_input_size += opt['ner_dim']

        # RNN document encoder
        with tf.variable_scope('document'):
            self.doc_rnn = layers.StackedBRNN(input_data=drnn_input, hidden_size=opt['hidden_size'], num_layers=opt['doc_layers'],dropout_rate=opt['dropout_rnn'])
            doc_hiddens = self.doc_rnn.output
            doc_hiddens = tf.concat([doc_hiddens[0], doc_hiddens[1]], axis=2)

        # RNN question encoder
        with tf.variable_scope('question'):
            self.question_rnn = layers.StackedBRNN(input_data=x2_emb, hidden_size=opt['hidden_size'], num_layers=opt['question_layers'],dropout_rate=opt['dropout_rnn'])
            # Encode question with RNN + merge hiddens
            question_hiddens = self.question_rnn.output
            question_hiddens = tf.concat([question_hiddens[0], question_hiddens[1]], axis=2)

            #q_merge_weights = layers.uniform_weights(question_hiddens, x2_mask)
            #question_hidden = layers.weighted_avg(question_hiddens, q_merge_weights)

            q_weight_layer = layers.LinearSeqAttn(question_hiddens, x2_mask)

            #question_hidden = tf.reduce_mean(question_hiddens, 1)
            question_hidden = q_weight_layer.weighted

        # Output sizes of rnn encoders
        doc_hidden_size = 2 * opt['hidden_size']
        question_hidden_size = 2 * opt['hidden_size']
        if opt['concat_rnn_layers']:
            doc_hidden_size *= opt['doc_layers']
            question_hidden_size *= opt['question_layers']

        # Bilinear attention for span start/end
        with tf.variable_scope("span_start"):
            self.start_attn = layers.BilinearSeqAttn(doc_hidden_size, question_hidden_size,doc_hiddens, question_hidden, x1_mask)
            self.start_scores = self.start_attn.alpha

        with tf.variable_scope("span_end"):
            self.end_attn = layers.BilinearSeqAttn(doc_hidden_size, question_hidden_size, doc_hiddens, question_hidden, x1_mask)
            self.end_scores = self.end_attn.alpha



