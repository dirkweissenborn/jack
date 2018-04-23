import json

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

FLAGS = tf.app.flags.FLAGS

reader = readers.reader_from_file(FLAGS.reader, assertion_dir=FLAGS.assertion_store)

input_module = reader.input_module

dataset = loaders[FLAGS.loader](FLAGS.dataset)

id2sideinformation = {}
bar = progressbar.ProgressBar(
    max_value=len(dataset),
    widgets=[' [', progressbar.Timer(), '] ', progressbar.Bar(), ' (', progressbar.ETA(), ') '])
for instance in bar(dataset):
    processed = input_module([instance[0]])
    assertions = processed[AssertionMRPorts.assertions]
    assertion_lengths = processed[AssertionMRPorts.assertion_lengths]
    rev_vocab = processed['__rev_vocab']
    a_strings = []

    for i in range(assertions.shape[0]):
        try:
            a_strings.append(' '.join(rev_vocab[idx] for idx in assertions[i, :assertion_lengths[i]]))
        except Exception:
            pass
    id2sideinformation[instance[0].id] = {'conceptnet': a_strings}
    if DefinitionPorts.definitions in processed:
        definitions = processed[DefinitionPorts.definitions]
        definition_lengths = processed[DefinitionPorts.definition_lengths]
        d_strings = []
        for i in range(definitions.shape[0]):
            d_strings.append(' '.join(rev_vocab[idx] for idx in definitions[i, :definition_lengths[i]]))
        id2sideinformation[instance[0].id]['wikipedia'] = d_strings

with open(FLAGS.output, 'w') as f:
    json.dump(id2sideinformation, f, sort_keys=True, indent=2, separators=(',', ': '))
