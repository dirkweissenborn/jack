"""Implementation of loaders for common datasets."""

import json

from jack.core.data_structures import *
from jack.io.SNLI2jtr import convert_snli
from jack.io.SQuAD2jtr import convert_squad

loaders = dict()


def _register(name):
    def _decorator(f):
        loaders[name] = f
        return f

    return _decorator


@_register('jack')
def load_jack(path, max_count=None):
    """
    This function loads a jack json file from a specific location.
    Args:
        path: the location to load from.
        max_count: how many instances to load at most

    Returns:
        A list of input-answer pairs.

    """
    # We load json directly instead
    with open(path) as f:
        jtr_data = json.load(f)

    return jack_to_qasetting(jtr_data, max_count)


@_register('squad')
def load_squad(path, max_count=None):
    """
    This function loads a squad json file from a specific location.
    Args:
        path: the location to load from.
        max_count: how many instances to load at most

    Returns:
        A list of input-answer pairs.
    """
    # We load to jtr dict and convert to qa settings for now
    jtr_data = convert_squad(path)
    return jack_to_qasetting(jtr_data, max_count)


@_register('snli')
def load_snli(path, max_count=None):
    """
    This function loads a jack json file with labelled answers from a specific location.
    Args:
        path: the location to load from.
        max_count: how many instances to load at most

    Returns:
        A list of input-answer pairs.
    """
    # We load to jtr dict and convert to qa settings for now
    jtr_data = convert_snli(path)
    return jack_to_qasetting(jtr_data, max_count)


@_register('assertion')
def load_assertions(path, max_count=None):
    qa_settings = []
    with open(path, 'r') as f:
        for l in f:
            if not l:
                continue
            l = l.strip()
            assertion, subj_span, obj_span = l.split('\t')
            subj_start, subj_end = subj_span.split(':')
            obj_start, obj_end = obj_span.split(':')
            subj = assertion[int(subj_start):int(subj_end)]
            obj = assertion[int(obj_start):int(obj_end)]
            if not subj or not obj:
                continue
            predicate = assertion[:int(subj_start)] + assertion[int(subj_end):int(obj_start)] + assertion[int(obj_end):]
            qa_settings.append((QASetting(subj + '|' + obj), [Answer(predicate.strip())]))
            if len(qa_settings) == max_count:
                break
    return qa_settings
