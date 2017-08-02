import re, time
import random
import string
from collections import Counter

import tensorflow as tf
import numpy as np
import chardet

import msgpack
import pandas as pd

data_folder ='../../data/SQuAD/'

def get_max_len(dt1, dt2=None):
    len1 = max(len(x) for x in dt1)
    if dt2 is None:
        return len1

    len2 = max(len(x) for x in dt2)

    if len1 > len2:
        return len1
    return len2

def load_data(folder=data_folder):
    opt = {}
    with open(folder+"meta.msgpack", 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = meta['embedding']

    opt['pretrained_words'] = True
    opt['vocab_size'] = len(embedding)
    opt['embedding_dim'] = len(embedding[0])

    with open(folder+"data.msgpack", 'rb') as f:
        data = msgpack.load(f, encoding='utf8')

    with open(folder+ 'dev.csv', 'rb') as f:
        charResult = chardet.detect(f.read())

    train_orig = pd.read_csv(folder+ 'train.csv', encoding=charResult['encoding'])
    dev_orig = pd.read_csv(folder+'dev.csv', encoding=charResult['encoding'])

    train = list(zip(
        data['trn_context_ids'],data['trn_context_features'],
        data['trn_context_tags'],data['trn_context_ents'],data['trn_question_ids'],
        train_orig['answer_start_token'].tolist(), train_orig['answer_end_token'].tolist(),
        data['trn_context_text'],data['trn_context_spans']
    ))
    dev = list(zip(
        data['dev_context_ids'],data['dev_context_features'],data['dev_context_tags'],
        data['dev_context_ents'],data['dev_question_ids'],data['dev_context_text'],data['dev_context_spans']
    ))
    dev_y = dev_orig['answers'].tolist()[:len(dev)]
    dev_y = [eval(y) for y in dev_y]

    # discover lengths
    opt['context_len'] = get_max_len(data['trn_context_ids'], data['dev_context_ids'])
    opt['feature_len'] = get_max_len(data['trn_context_features'][0], data['dev_context_features'][0])
    opt['question_len'] = get_max_len(data['trn_question_ids'], data['dev_question_ids'])

    print(train_orig['answer_start_token'].tolist()[:10])

    return train, dev, dev_y, embedding, opt

def _normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def _exact_match(pred, answers):
    if pred is None or answers is None:
        return False
    pred = _normalize_answer(pred)
    for a in answers:
        if pred == _normalize_answer(a):
            return True
    return False


def _f1_score(pred, answers):
    def _score(g_tokens, a_tokens):
        common = Counter(g_tokens) & Counter(a_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1. * num_same / len(g_tokens)
        recall = 1. * num_same / len(a_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    if pred is None or answers is None:
        return 0
    g_tokens = _normalize_answer(pred).split()
    scores = [_score(g_tokens, _normalize_answer(a).split()) for a in answers]
    return max(scores)


def score(pred, truth):
    assert len(pred) == len(truth)
    f1 = em = total = 0
    for p, t in zip(pred, truth):
        total += 1
        em += _exact_match(p, t)
        f1 += _f1_score(p, t)
    em = 100. * em / total
    f1 = 100. * f1 / total
    return em, f1

class BatchGen:
    def __init__(self, data, batch_size, opt, evaluation=False):
        self.batch_size = batch_size
        self.eval = evaluation
        self.opt = opt

        if not evaluation:
            indices = list(range(len(data)))
            random.shuffle(indices)
            data = [data[i] for i in indices]
        # chunk into batches
        data = [data[i:i + batch_size] for i in range(0, len(data), batch_size)]
        self.data = data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        for batch in self.data:
            batch_size = len(batch)
            batch = list(zip(*batch))
            if self.eval:
                assert len(batch) == 7
            else:
                assert len(batch) == 9

            context_len = self.opt['context_len']
            context_id = np.zeros((batch_size, context_len))
            for i, doc in enumerate(batch[0]):
                context_id[i, :len(doc)] = doc

            feature_len = len(batch[1][0][0])
            context_feature = np.zeros((batch_size, context_len, feature_len))
            for i, doc in enumerate(batch[1]):
                for j, feature in enumerate(doc):
                    context_feature[i, j, :] = feature

            context_tag = np.zeros((batch_size, context_len))
            for i, doc in enumerate(batch[2]):
                context_tag[i, :len(doc)] = doc

            context_ent = np.zeros((batch_size, context_len))
            for i, doc in enumerate(batch[3]):
                context_ent[i, :len(doc)] = doc

            question_len =self.opt['question_len']
            question_id = np.zeros((batch_size, question_len))
            for i, doc in enumerate(batch[4]):
                question_id[i, :len(doc)] = doc

            context_mask = np.equal(context_id, 0)
            question_mask = np.equal(question_id, 0)
            if not self.eval:
                y_s = batch[5]
                y_e = batch[6]

            text = list(batch[-2])
            span = list(batch[-1])
            if self.eval:
                yield (context_id, context_feature, context_tag, context_ent, context_mask,question_id, question_mask, text, span)
            else:
                yield (context_id, context_feature, context_tag, context_ent, context_mask, question_id, question_mask, y_s, y_e, text, span)

t_initial = time.time()
print("Loading data...")
train, dev, dev_y, embedding, opt = load_data()
print("Loading checkpoint...")
checkpoint_file = tf.train.latest_checkpoint('../summary/1501541746.9058044')

graph = tf.Graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        print("Restoring session...")
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)
        print("Begin predicting")

        tf_doc_words = graph.get_operation_by_name("document_words").outputs[0]
        tf_word_feat = graph.get_operation_by_name("word_features").outputs[0]
        tf_pos = graph.get_operation_by_name("pos_features").outputs[0]
        tf_ner = graph.get_operation_by_name("ner_features").outputs[0]
        tf_doc_mask = graph.get_operation_by_name("doc_mask").outputs[0]
        tf_q_words = graph.get_operation_by_name("question_words").outputs[0]
        tf_q_mask = graph.get_operation_by_name("question_mask").outputs[0]

        target_start = graph.get_operation_by_name("span_start/BilinearSeqAttention/alpha").outputs[0]
        target_end = graph.get_operation_by_name("span_end/BilinearSeqAttention/alpha").outputs[0]

        batches = BatchGen(dev, batch_size=128, opt=opt, evaluation=True)
        predictions = []
        batch_count = 0

        for batch in batches:
            t_start = time.time()

            feed_dict = {
                tf_doc_words: batch[0], tf_word_feat: batch[1], tf_pos: batch[2],
                tf_ner: batch[3], tf_doc_mask: batch[4], tf_q_words: batch[5], tf_q_mask: batch[6]
            }

            ops = [target_start, target_end]

            sc_s, sc_e = sess.run(ops, feed_dict=feed_dict)

            # Get argmax text spans
            text = batch[-2]
            spans = batch[-1]

            max_len = len(sc_s[0])

            for i in range(len(sc_s)):
                scores = np.outer(sc_s[i], sc_e[i])
                scores = np.tril(np.triu(scores), max_len-1)
                #scores = scores.triu().tril(max_len - 1)

                s_idx, e_idx = np.unravel_index(np.argmax(scores), scores.shape)
                if s_idx < len(spans[i]) and e_idx < len(spans[i]):
                    s_offset = spans[i][s_idx][0]
                    e_offset = spans[i][e_idx][1]

                    if s_offset<e_offset:
                        predictions.append(text[i][s_offset:e_offset])
                    else:
                        predictions.append("<NA>")
                #st = [sp for sp in spans[i] if sp[0]<s_idx and sp[1]>=s_idx]
                #ed = [sp for sp in spans[i] if sp[0] <= e_idx and sp[1] > e_idx]
                #if len(st)>0 and len(ed)>0:
                #    s_offset = st[0][0]
                #    e_offset = ed[0][1]

                    #s_offset, e_offset = spans[i][s_idx][0], spans[i][e_idx][1]
                #    predictions.append(text[i][s_offset:e_offset])
                else:
                    predictions.append("<NA>")

                print("batch {}: {} questions in {} seconds".format(batch_count, len(sc_s), time.time() - t_start))

            batch_count += 1
        em, f1 = score(predictions, dev_y)

        print("dev EM: {} F1: {}".format(em, f1))

dt = list(map(lambda x,y:[x,y[0]],predictions,dev_y))

df = pd.DataFrame(dt,columns=["pred","label"])

df.to_excel("../predictions.xlsx", index=False)

print("Total script time: {} seconds".format(time.time() - t_initial))




