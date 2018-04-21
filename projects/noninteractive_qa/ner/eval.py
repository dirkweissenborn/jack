import math

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from jack import readers
from jack.core import QASetting
import tensorflow as tf
import numpy as np
import logging


PATH_TO_SENTEVAL = './SentEval'


TRAIN = 'projects/noninteractive_qa/ner/data/eng.testb'
DEV = 'projects/noninteractive_qa/ner/data/eng.testa'
TEST = 'projects/noninteractive_qa/ner/data/eng.testb'

BATCH_SIZE = 32

def load_ner(file, labels=None):
    docs = list()
    annot = list()
    labels = labels or {}
    with open(file, 'rb') as f:
        for l in f:
            l = l.decode('utf-8').strip()
            if l.startswith('-DOCSTART-'):
                docs.append([])
                annot.append([])
            elif l:
                split = l.split()
                docs[-1].append(split[0])
                if split[3] not in labels:
                    labels[split[3]] = len(labels)
                annot[-1].append(labels[split[3]])
    return docs, annot, labels


def tensorize_data(reader, docs, annot):
    g = tf.get_default_graph()
    question_op = [op.outputs[0] for op in g.get_operations() if op.name.endswith('question_representation')][0]

    embedded = []
    for i in range(int(math.ceil(len(docs) / BATCH_SIZE))):
        idx = i * BATCH_SIZE
        batch = docs[idx:idx + BATCH_SIZE]
        batch = [sent if sent != [] else ['.'] for sent in batch]
        batch = [QASetting(' '.join(sent), ['']) for sent in batch]

        tensors = reader.input_module(batch)
        encoded = reader.session.run(question_op, feed_dict=reader.model_module.convert_to_feed_dict(tensors))
        for k in range(encoded.shape[0]):
            embedded.append(encoded[k, :len(docs[idx + k])])
    embedded = np.concatenate(embedded, 0)
    annot = np.concatenate(annot, 0)
    return embedded, annot

logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == '__main__':
    import sys

    sys.path.insert(0, PATH_TO_SENTEVAL)
    from senteval.tools.validation import SplitClassifier

    reader_dir = sys.argv[1]

    logging.info('Loading Reader')
    reader = readers.reader_from_file(reader_dir)

    labels = {'O': 0}

    logging.info('Loading Data')
    docs_test, annot_test, _ = load_ner(TEST, labels)
    label_list = [None] * len(labels)
    for k, v in labels.items():
        label_list[v] = k

    docs_train, annot_train, _ = load_ner(TRAIN, labels)
    docs_dev, annot_dev, _ = load_ner(DEV, labels)

    logging.info('Encoding Data')
    X, y = {}, {}
    X["train"], y["train"] = tensorize_data(reader, docs_train, annot_train)
    X["valid"], y["valid"] = tensorize_data(reader, docs_dev, annot_dev)
    X["test"], y["test"] = tensorize_data(reader, docs_test, annot_test)

    print('train:', X['train'].shape[0], 'valid:', X['valid'].shape[0], 'test:', X['test'].shape[0])

    logging.info('Training')
    regs = [2 ** t for t in range(-2, 4, 1)]
    regs = [1.0]
    scores = []
    clfs = []
    seed = 123
    for reg in regs:
        clf = LogisticRegression(C=reg, random_state=seed)
        clf.fit(X['train'], y['train'])
        scores.append(round(100 * clf.score(X['valid'], y['valid']), 2))
        clfs.append(clf)

    logging.info([('reg:' + str(regs[idx]), scores[idx])
                  for idx in range(len(scores))])
    optreg = regs[np.argmax(scores)]
    devacc = np.max(scores)
    logging.info('Validation : best param found is reg = {0} with score \
                {1}'.format(optreg, devacc))
    clf = clfs[np.argmax(scores)]

    y_pred = clf.predict(X['test'])

    print(classification_report(y['test'], y_pred, target_names=label_list))

