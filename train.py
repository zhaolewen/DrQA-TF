import re,os,sys, time
import random
import string
import logging
import argparse
from shutil import copyfile
from datetime import datetime
from collections import Counter
import requests
import tensorflow as tf
import numpy as np
import chardet
import msgpack
import pandas as pd
from drqa.model import DocReaderModel
#from tensorflow.python import debug as tf_debug

parser = argparse.ArgumentParser(description='Train a Document Reader model.')
# system
parser.add_argument('--log_file', default='output.log',help='path for log file.')
parser.add_argument('--log_per_updates', type=int, default=3, help='log model loss per x updates (mini-batches).')
parser.add_argument('--data_file', default='./SQuAD/data.msgpack',help='path to preprocessed data file.')
parser.add_argument('--model_dir', default='./summary/',help='path to store saved models.')
parser.add_argument('--save_last_only', action='store_true',help='only save the final models.')
parser.add_argument('--eval_per_epoch', type=int, default=1,help='perform evaluation per x epoches.')
parser.add_argument('--eval_per_step', type=int, default=500,help='perform evaluation per x step.')
parser.add_argument('--squad_dir', default='./SQuAD/',help='directory for SQuAD files')
# training
parser.add_argument('-e', '--epoches', type=int, default=20)
parser.add_argument('-bs', '--batch_size', type=int, default=32)
parser.add_argument('-rs', '--resume', default=None,help='previous model file name (in `model_dir`). e.g. "checkpoint_epoch_11.pt"')
parser.add_argument('-gc', '--grad_clipping', type=float, default=10)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
parser.add_argument('-ld', '--learning_decay', type=float, default=0.96, help="decay rate of learning rate for every 1000 steps")
parser.add_argument('-tp', '--tune_partial', type=int, default=1000,help='finetune top-x embeddings.')
parser.add_argument('--fix_embeddings', action='store_true',help='if true, `tune_partial` will be ignored.')
# model
parser.add_argument('--doc_layers', type=int, default=1)
parser.add_argument('--question_layers', type=int, default=1)
parser.add_argument('--hidden_size', type=int, default=128)
parser.add_argument('--num_features', type=int, default=4)
parser.add_argument('--pos', type=bool, default=True)
parser.add_argument('--pos_size', type=int, default=56, help='how many kinds of POS tags.')
parser.add_argument('--pos_dim', type=int, default=12, help='the embedding dimension for POS tags.')
parser.add_argument('--ner', type=bool, default=True)
parser.add_argument('--ner_size', type=int, default=19, help='how many kinds of named entity tags.')
parser.add_argument('--ner_dim', type=int, default=8, help='the embedding dimension for named entity tags.')
parser.add_argument('--use_qemb', type=bool, default=False)
parser.add_argument('--concat_rnn_layers', type=bool, default=True)
parser.add_argument('--dropout_rnn', type=float, default=0.7)
parser.add_argument('--max_len', type=int, default=15)

args = parser.parse_args()

# set model dir
model_dir = args.model_dir
if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)
model_dir = os.path.abspath(model_dir)

# setup logger
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
fh = logging.FileHandler(args.log_file)
fh.setLevel(logging.DEBUG)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter(fmt='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S')
fh.setFormatter(formatter)
ch.setFormatter(formatter)
log.addHandler(fh)
log.addHandler(ch)

def sendStatElastic(data, endpoint="http://52.48.27.79:9200/neural/testnn"):
    data['step_time'] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    requests.post(endpoint, json=data)


def main():
    log.info('[program starts.]')
    train, dev, dev_y, embedding, opt = load_data(vars(args))
    log.info('[Data loaded.]')

    run_name = str(time.time())
    out_dir = opt["model_dir"] + run_name + "/"

    graph = tf.Graph()
    with graph.as_default():
        log.info('[Loading graph.]')
        model = DocReaderModel(opt, embedding)
        log.info('[Graph loaded.]')

        sv = tf.train.Supervisor(logdir=opt["model_dir"])

        with sv.managed_session() as sess:

            epoch_0 = 1

            #saver = tf.train.Saver(tf.global_variables())
            #sess.run(tf.global_variables_initializer())

            log.info('[Begin training.]')
            step = 0
            test_count = 0
            for epoch in range(epoch_0, epoch_0 + args.epoches):
                log.warning('Epoch {}'.format(epoch))
                # train
                batches = BatchGen(train, batch_size=args.batch_size, opt=opt)
                start = datetime.now()

                for i, batch in enumerate(batches):
                    step, tr_summary, _, loss, preds, y_true, learn_rate = model.train(batch, sess)

                    em, f1 = score(preds, y_true)
                    log.warning("train EM: {} F1: {}".format(em, f1))
                    sendStatElastic({"phase":"train","name":"DrQA","run_name":run_name,"step":int(step),"precision":float(em),"f1":float(f1),"loss":float(loss),"epoch":epoch, "learning_rate":float(learn_rate)})

                    if i % args.log_per_updates == 0:
                        log.info('updates[{}]  remaining[{}]'.format(step,str((datetime.now() - start) / (i + 1) * (len(batches) - i - 1)).split('.')[0]))

                    # eval
                    if step - test_count* args.eval_per_step > args.eval_per_step:
                        te_batches = BatchGen(dev, batch_size=args.batch_size, opt=opt, evaluation=True)
                        predictions = []
                        for batch in te_batches:
                            predictions.extend(model.test(batch, sess))
                        em, f1 = score(predictions, dev_y)
                        sendStatElastic({"phase": "test", "name": "DrQA", "run_name": run_name, "step": float(step),"precision": float(em), "f1": float(f1), "epoch": epoch})
                        log.warning("dev EM: {} F1: {}".format(em, f1))
                        test_count += 1

                #if epoch % args.eval_per_epoch == 0:
                #    save_path = saver.save(sess, out_dir + "model_max.ckpt")
                #    print("max model saved in file: %s" % save_path)


def get_max_len(dt1, dt2=None):
    len1 = max(len(x) for x in dt1)
    if dt2 is None:
        return len1

    len2 = max(len(x) for x in dt2)

    if len1 > len2:
        return len1
    return len2

def load_data(opt):
    with open(opt["squad_dir"]+'meta.msgpack', 'rb') as f:
        meta = msgpack.load(f, encoding='utf8')
    embedding = meta['embedding']

    opt['pretrained_words'] = True
    opt['vocab_size'] = len(embedding)
    opt['embedding_dim'] = len(embedding[0])

    with open(args.data_file, 'rb') as f:
        data = msgpack.load(f, encoding='utf8')

    with open(opt["squad_dir"]+ 'train.csv', 'rb') as f:
        charResult = chardet.detect(f.read())

    train_orig = pd.read_csv(opt["squad_dir"]+ 'train.csv', encoding=charResult['encoding'])
    dev_orig = pd.read_csv(opt["squad_dir"]+'dev.csv', encoding=charResult['encoding'])

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
    pred = _normalize_answer(pred)
    if pred is None or answers is None:
        return False
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

if __name__ == '__main__':
    main()
