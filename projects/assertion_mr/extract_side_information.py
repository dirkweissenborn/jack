import json
import math

import progressbar
import tensorflow as tf

from jack import readers
from jack.io.load import loaders
from projects.assertion_mr.qa.definition_model import DefinitionPorts
from projects.assertion_mr.shared import AssertionMRPorts

tf.app.flags.DEFINE_string('dataset', None, 'dataset file.')
tf.app.flags.DEFINE_string('output', None, 'output json.')
tf.app.flags.DEFINE_string('loader', 'squad', 'loader type.')
tf.app.flags.DEFINE_string('assertion_store', None, 'assertion_store.')
tf.app.flags.DEFINE_string('reader', None, 'path to reader.')
tf.app.flags.DEFINE_integer('batch_size', 64, 'batch_size.')

FLAGS = tf.app.flags.FLAGS
dataset = loaders[FLAGS.loader](FLAGS.dataset)
if FLAGS.assertion_store:
    reader = readers.reader_from_file(FLAGS.reader, assertion_dir=FLAGS.assertion_store)
else:
    reader = readers.reader_from_file(FLAGS.reader)

input_module = reader.input_module

num_batches = int(math.ceil(len(dataset) / FLAGS.batch_size))

id2sideinformation = {}
bar = progressbar.ProgressBar(
    max_value=num_batches,
    widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') '])

for idx in bar(range(0, len(dataset), FLAGS.batch_size)):
    instances = dataset[idx:idx + FLAGS.batch_size]
    processed = input_module(instances)
    assertions = processed[AssertionMRPorts.assertions]
    assertion_lengths = processed[AssertionMRPorts.assertion_lengths]
    assertion2question = processed[AssertionMRPorts.assertion2question]
    rev_vocab = processed['__rev_vocab']
    a_strings = [[] for _ in instances]

    for i in range(assertions.shape[0]):
        b_idx = assertion2question[i]
        try:
            a_strings[b_idx].append(' '.join(rev_vocab[idx] for idx in assertions[i, :assertion_lengths[i]]))
        except Exception:
            pass
    for instance, a_s in zip(instances, a_strings):
        id2sideinformation[instance[0].id] = {'conceptnet': a_s}
    if DefinitionPorts.definitions in processed:
        definitions = processed[DefinitionPorts.definitions]
        definition_lengths = processed[DefinitionPorts.definition_lengths]
        definition2question = processed[DefinitionPorts.definition2question]
        d_strings = [[] for _ in instances]
        for i in range(definitions.shape[0]):
            b_idx = definition2question[i]
            d_strings[b_idx].append(' '.join(rev_vocab[idx] for idx in definitions[i, :definition_lengths[i]]))
        for instance, d_s in zip(instances, d_strings):
            id2sideinformation[instance[0].id]['wikipedia'] = d_s

with open(FLAGS.output, 'w') as f:
    json.dump(id2sideinformation, f, sort_keys=True, indent=2, separators=(',', ': '))
