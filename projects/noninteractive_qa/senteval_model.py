# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from __future__ import absolute_import, division, unicode_literals

import sys
import numpy as np
import logging

import data
import tensorflow as tf
from jack import readers

# Set PATHs
from jack.core import QASetting

PATH_TO_SENTEVAL = './SentEval'
PATH_TO_DATA = 'SentEval/data/senteval_data'
PATH_TO_READER = './multirep_reader'
REDUCTION_METHOD = 'max'

# import SentEval
sys.path.insert(0, PATH_TO_SENTEVAL)
import senteval


reader = readers.reader_from_file(PATH_TO_READER)
g = tf.get_default_graph()
question_op = [op.outputs[0] for op in g.get_operations() if op.name.endswith('question_representation')][0]

def prepare(params, samples):
    return

def batcher(params, batch):
    batch = [sent if sent != [] else ['.'] for sent in batch]
    batch = [QASetting(' '.join(sent), ['']) for sent in batch]

    tensors = reader.input_module(batch)
    embeddings = reader.session.run(question_op, feed_dict=reader.model_module.convert_to_feed_dict(tensors))
    if REDUCTION_METHOD == 'max':
        embeddings = np.max(embeddings, 1)
    else:
        embeddings = np.mean(embeddings, 1)

    return embeddings


# Set params for SentEval
params_senteval = {'task_path': PATH_TO_DATA, 'usepytorch': False, 'kfold': 10}
params_senteval['classifier'] = {'nhid': 0, 'optim': 'adam', 'batch_size': 64,
                                 'tenacity': 5, 'epoch_size': 4}

# Set up logger
logging.basicConfig(format='%(asctime)s : %(message)s', level=logging.DEBUG)

if __name__ == "__main__":
    se = senteval.engine.SE(params_senteval, batcher, prepare)
    transfer_tasks = ['STS12', 'STS13', 'STS14', 'STS15', 'STS16',
                      'MR', 'CR', 'MPQA', 'SUBJ', 'SST2', 'SST5', 'TREC', 'MRPC',
                      'SICKEntailment', 'SICKRelatedness', 'STSBenchmark']
    results = se.eval(transfer_tasks)
    print(results)
