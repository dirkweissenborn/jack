import json


def load_sentences(path):
    sentences = {}
    with open(path, 'r') as f:
        for l in f:
            l = l.rstrip()
            split = l.split('\t')
            sentences[split[0]] = (split[1], split[1].split(' '))

    return sentences


def token2char_idx(token_idx, tokens):
    return sum(len(t) + 1 for t in tokens[:token_idx])


def load(dataset_path, sentences_path):
    sentence_dict = load_sentences(sentences_path)
    qas = []
    qa_dict = {}
    with open(dataset_path, 'r') as f:
        for l in f:
            split = l.rstrip().split('\t')
            sent_id = split[0]
            sentence, sentence_tokens = sentence_dict[sent_id]
            spans = []
            for ans in split[7:]:
                a_split = ans.split(':')[1].split()
                start = token2char_idx(int(a_split[0]), sentence_tokens)
                end = token2char_idx(int(a_split[-1]) + 1, sentence_tokens) - 1
                spans.append([start, end])

            if sent_id not in qa_dict:
                qa = {
                    "support": [sentence],
                    "questions": [{
                        'question': {
                            'text': split[5],
                            'id': '_'.join(split[:4])
                        },
                        'answers': [{"span": span, "text": sentence[span[0]:span[1]]} for span in spans]
                    }]
                }
                qas.append(qa)
                qa_dict[sent_id] = qa
            else:
                qa = qa_dict[sent_id]
                qa["questions"].append({
                    'question': {
                        'text': split[5],
                        'id': '_'.join(split[:4])
                    },
                    'answers': [{"span": span, "text": sentence[span[0]:span[1]]} for span in spans]
                })
    return {
        'meta': dataset_path.split('/')[-1],
        'instances': qas
    }


def main():
    import sys
    data = load(sys.argv[1], sys.argv[2])
    with open(sys.argv[3], 'w') as outfile:
        json.dump(data, outfile, indent=2)


if __name__ == "__main__":
    main()
