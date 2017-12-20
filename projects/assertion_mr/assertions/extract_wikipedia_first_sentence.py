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
            assertion = abstract[:sentence_end]
        else:
            assertion = abstract
        subjects = []
        for l in labels.get(article, [uncamel(article).replace('_', ' ')]):
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
    tf.app.flags.DEFINE_string('dbpedia_transitive_types', None, 'path to dbpedia instance transitive types')
    tf.app.flags.DEFINE_string('assertion_store_path', None, 'directory to assertion store')

    FLAGS = tf.app.flags.FLAGS
    store = AssertionStore(FLAGS.assertion_store_path, True)

    logger.info('Loading DBpedia types')
    allowed = set()
    with BZ2File(FLAGS.dbpedia_transitive_types) as f:
        for l in f:
            l = l.decode('utf-8')
            if l.startswith('#'):
                continue
            split = l.split('> ')
            subj = split[0][split[0].rindex('/') + 1:]
            if subj.startswith('List_of') or '__' in subj or subj in allowed:
                continue
            obj = split[2]
            obj = obj[obj.rindex('/') + 1:]
            if obj != 'owl#Thing':  # we only consider entities that have some other type than only owl#Thing
                allowed.add(subj)
    logger.info('Number of allowed entities: %d' % len(allowed))


    def simple_parse(fn, allowed_subj=None, allowed_obj=None):
        d = defaultdict(list)
        with BZ2File(fn) as f:
            for l in f:
                l = l.decode('utf-8')
                if l.startswith('#'):
                    continue
                split = l.split('> ')
                subj = split[0][split[0].rindex('/') + 1:]
                if allowed_subj is not None and subj not in allowed_subj:
                    continue
                obj = split[2]
                if obj.startswith('"'):
                    obj = obj[1:]
                    obj = obj[:obj.find('"@en')]
                else:
                    obj = obj[obj.rindex('/') + 1:]
                if allowed_obj is not None and obj not in allowed_obj:
                    continue
                d[subj].append(obj)
        return d

    logger.info('Loading DBpedia labels')
    labels = simple_parse(FLAGS.dbpedia_labels)

    # reg = r'( )?\([^)]+\)'
    # for k, vs in labels.items():
    #    for v in vs:
    #        if '(' in v:
    #            labels[k].append(re.sub(reg, '', v))

    logger.info('Loading DBpedia abstracts')
    abstracts = simple_parse(FLAGS.dbpedia_short_abstracts, allowed_subj=allowed)
    logger.info('Loading DBpedia disambiguations')
    disambiguations = simple_parse(FLAGS.dbpedia_disambiguates, allowed_obj=allowed)
    logger.info('Loading DBpedia redirects')
    transitive_redirects = simple_parse(FLAGS.dbpedia_transitive_redirect, allowed_obj=allowed)

    logger.info('Extending DBpedia labels with redirects and disambiguations')
    labels_ext = dict()
    labels_ext.update((k, set(vs)) for k, vs in labels.items() if k in allowed)
    for k, vs in transitive_redirects.items():
        for v in vs:
            if v in labels_ext:
                labels_ext[v].add(labels.get(k, [uncamel(k).replace('_', ' ')])[0])
    for k, vs in disambiguations.items():
        k = k.replace('_(disambiguation)', '')
        for v in vs:
            if v in labels_ext:
                labels_ext[v].add(labels.get(k, [uncamel(k).replace('_', ' ')])[0])

    logger.info('Writing first wikipedia sentences as assertions...')
    extract_assertions(abstracts, labels_ext, store)
    store.save()
