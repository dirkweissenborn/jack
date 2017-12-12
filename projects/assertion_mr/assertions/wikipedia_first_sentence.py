import re
from bz2 import BZ2File
from collections import defaultdict

import spacy

from projects.assertion_mr.assertions.store import AssertionStore


def uncamel(name):
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1 \2', name)
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', s1).lower()


def extract_assertions(abstracts, labels, store):
    # nlp = spacy.load('en', disable=['parser', 'ner', 'textcat'])
    nlp = spacy.load('en', parser=False)

    lemma_labels = dict()
    counter = 0
    for article, abstract in abstracts.items():
        abstract = abstract[0]
        if counter % 100000 == 0:
            logger.info('%d assertions added' % counter)
        sentence_end = abstract.find('. ')
        if sentence_end > 0:
            assertion = abstract[sentence_end]
        else:
            assertion = abstract
        subjects = []
        for l in labels[article]:
            if l not in lemma_labels:
                lemma_labels[l] = ' '.join(t.lemma_ for t in nlp(l))
            subjects.append(lemma_labels[l])
        store.add_assertion(assertion, subjects, [], 'wikipedia_firstsent')
        counter += 1


if __name__ == '__main__':
    import logging
    import os
    import sys
    import tensorflow as tf

    logger = logging.getLogger(os.path.basename(sys.argv[0]))
    logging.basicConfig(level=logging.INFO)

    tf.app.flags.DEFINE_string('dbpedia_short_abstracts', None, 'path to dbpedia short abstracts')
    tf.app.flags.DEFINE_string('dbpedia_labels', None, 'path to dbpedia labels')
    tf.app.flags.DEFINE_string('dbpedia_disambiguates', None, 'path to dbpedia disambiguations')
    tf.app.flags.DEFINE_string('dbpedia_transitive_redirect', None, 'path to dbpedia transitive redirects')
    tf.app.flags.DEFINE_string('assertion_store_path', None, 'directory to assertion store')

    FLAGS = tf.app.flags.FLAGS
    store = AssertionStore(FLAGS.assertion_store_path, True)


    def simple_parse(fn):
        d = defaultdict(list)
        with BZ2File(fn) as f:
            for l in f:
                l = l.decode('utf-8')
                if l.startswith('#'):
                    continue
                split = l.split('> ')
                subj = split[0][split[0].rindex('/') + 1:]
                obj = split[2]
                if obj.startswith('"'):
                    obj = obj[1:]
                    obj = obj[:obj.find('"@en')]
                else:
                    obj = obj[obj.rindex('/') + 1:]
                d[subj].append(obj)
        return d


    logger.info('Loading DBpedia Data')
    labels = simple_parse(FLAGS.dbpedia_labels)
    abstracts = simple_parse(FLAGS.dbpedia_short_abstracts)
    disambiguations = simple_parse(FLAGS.dbpedia_disambiguates)
    transitive_redirects = simple_parse(FLAGS.dbpedia_transitive_redirect)

    for k, vs in disambiguations.items():
        for v in vs:
            labels[v].append(labels.get(k, [uncamel(k).replace('_', ' ')])[0])
    for k, vs in disambiguations.items():
        for v in vs:
            labels[k].append(labels.get(v, [uncamel(v).replace('_', ' ')])[0])

    logger.info('Writing first wikipedia sentences as assertions...')
    extract_assertions(abstracts, labels, store)
    store.save()
