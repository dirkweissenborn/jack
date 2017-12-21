import re
from bz2 import BZ2File
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import spacy

from projects.assertion_mr.assertions.store import AssertionStore


def uncamel(name):
    return re.sub('([a-z0-9])([A-Z])', r'\1 \2', name)


def write_assertions(abstracts, labels, store):
    # nlp = spacy.load('en', disable=['parser', 'ner', 'textcat'])
    nlp = spacy.load('en', parser=False)

    lemma_labels = dict()
    counter = 0
    reg = r'( )?\([^)]+\)'

    for article, abstracts in abstracts.items():
        for abstract in abstracts:
            if counter % 100000 == 0:
                logger.info('%d assertions added' % counter)
            sentence_end = abstract.find('. ')
            if sentence_end > 0:
                assertion = abstract[:sentence_end]
            else:
                assertion = abstract
            subjects = set()
            ll = labels.get(article)
            if ll is None or not ll:
                ll = [uncamel(article).replace('_', ' ')]
            for l in ll:
                if l not in lemma_labels:
                    lemma_labels[l] = ' '.join(t.lemma_ for t in nlp(l))
                subjects.add(lemma_labels[l])
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
    tf.app.flags.DEFINE_string('dbpedia_transitive_redirect', None, 'path to dbpedia transitive redirects')
    tf.app.flags.DEFINE_string('assertion_store_path', None, 'directory to assertion store')

    FLAGS = tf.app.flags.FLAGS


    def simple_parse(fn):
        d = defaultdict(set)
        with BZ2File(fn) as f:
            for l in f:
                l = l.decode('utf-8')
                if l.startswith('#'):
                    continue
                split = l.split('> ')
                subj = split[0][split[0].rindex('/') + 1:]
                if subj.startswith('List_of') or '(disambiguation)' in subj or '__' in subj:
                    continue
                obj = split[2]
                if obj.startswith('"'):
                    obj = obj[1:]
                    obj = obj[:obj.find('"@en')]
                else:
                    obj = obj[obj.rindex('/') + 1:]
                if not (obj.startswith('List_of') or '(disambiguation)' in obj or '__' in obj):
                    d[subj].add(obj)
        return d


    pool = ThreadPoolExecutor(4)
    logger.info('Loading DBpedia labels, redirects, abstracts...')
    result = [d for d in pool.map(
        simple_parse, [FLAGS.dbpedia_labels, FLAGS.dbpedia_transitive_redirect, FLAGS.dbpedia_short_abstracts])]
    labels, transitive_redirects, abstracts = result

    reg = r'( )?\([^)]+\)'
    for k, ll in labels.items():
        toadd = list()
        for l in ll:
            if '(' in l:
                toadd.append(re.sub(reg, '', l))
        ll.update(toadd)

    logger.info('Extending DBpedia labels with redirects...')
    for k, vs in transitive_redirects.items():
        for v in vs:
            labels[v].add(uncamel(k).replace('_', ' '))

    logger.info('Writing first wikipedia sentences for %d entities...' % len(abstracts))

    store = AssertionStore(FLAGS.assertion_store_path, writeback=True)
    write_assertions(abstracts, labels, store)
    store.save()
