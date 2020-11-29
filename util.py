import os
import fire
from collections import defaultdict
from tqdm import tqdm
import random
from random import randint, shuffle
import json
import gzip


def dummy_baseline_predict(
    hyponyms_file: str, hypernyms_file: str, positive_wcl_sentences: str,
    output_file: str
):
    hyponym2hypernym = defaultdict(set)

    hyponyms = [tuple(x.strip().lower().split('\t')) for x in open(hyponyms_file).readlines()]
    positive_wcl_sentences = open(positive_wcl_sentences).readlines()
    hypernyms_file = open(hypernyms_file).readlines()

    for i, line in tqdm(enumerate(positive_wcl_sentences), total=len(positive_wcl_sentences)):
        if i % 2 == 1:
            hyponym = line.strip().split(':')[0]
            hypernym = [
                x.split('_')[-1] for x in
                    line.split('</HYPER>')[0].split('<HYPER>')[-1].strip().lower().split()
            ]
            hyponym2hypernym[hyponym].add(' '.join(hypernym))

    with open(output_file, 'w') as p, open(output_file + '.gold.txt', 'w') as g:
        for i, (hyponym, hyponym_type) in tqdm(enumerate(hyponyms), total=len(hyponyms)):
            if hyponym in hyponym2hypernym:
                print('\t'.join(list(hyponym2hypernym[hyponym])), file=p)
            else:
                print('', file=p)
            print(hypernyms_file[i].strip(), file=g)


def extract_potential_definitional_sentences(
    hyponyms_file: str, hypernyms_file: str, index_file: str,
    text_corpus: str, output_file: str, only_hyponyms: bool = False, max_num_sents = None,
    total_lines=132498489
):
    # extract all sentences containing tokens of (hyponym, hypernym) pairs
    hyponyms = [line.rstrip().lower().split('\t')[0].split() for line in open(hyponyms_file).readlines()]
    hyponym_hypernyms = [
        [hypernym.split() for hypernym in line.rstrip().lower().split('\t')] for
            line in open(hypernyms_file).readlines()
    ]
    print(hyponyms[0])
    print(hyponym_hypernyms[0])
    assert len(hyponyms) == len(hyponym_hypernyms), f"len(hyponyms) = {len(hyponyms)}; len(hyponym_hypernyms) = {len(hyponym_hypernyms)}"
    print('loading index file')
    index = json.load(open(index_file))
    for token in tqdm(index, total=len(index), desc='index convertation ...'):
        index[token] = set(index[token])

    potential_sentences_ids = set()
    num_hypo_hyper_pairs_matches = 0
    not_found_hyponyms = []
    not_found_hypernyms = []

    for hyponym, hypernyms in tqdm(
        zip(hyponyms, hyponym_hypernyms),
        total=len(hyponyms),
        desc='extracting potential definitional sentences ...'
    ):
        if any([hyponym_token not in index for hyponym_token in hyponym]):
            not_found_hyponyms.append(' '.join(hyponym))
            continue
        hyponym_ids = index[hyponym[-1]].copy()
        for hyponym_token in hyponym[:-1]:
            hyponym_ids = hyponym_ids.intersection(index[hyponym_token])

        if only_hyponyms:
            hyponym_ids = list(hyponym_ids)
            shuffle(hyponym_ids)
            hyponym_ids = set(hyponym_ids[:max_num_sents])
            potential_sentences_ids = potential_sentences_ids.union(hyponym_ids)
            continue

        first_finded_hypernym = True
        for hypernym in hypernyms:
            if any([hypernym_token not in index for hypernym_token in hypernym]):
                not_found_hypernyms.append(' '.join(hypernym))
                continue
            cur_search_ids = hyponym_ids.copy()
            for hypernym_token in hypernym:
                cur_search_ids = cur_search_ids.intersection(index[hypernym_token])

            if first_finded_hypernym:
                first_finded_hypernym = False
                num_hypo_hyper_pairs_matches += 1

            potential_sentences_ids = potential_sentences_ids.union(cur_search_ids)

    print('found', len(potential_sentences_ids), 'potential definitional sentences')
    print('not found hyponyms:', len(not_found_hyponyms))
    print(*not_found_hyponyms, sep='$$$')
    print('not found hypernyms:', len(not_found_hypernyms))
    print(*not_found_hypernyms, sep='$$$')
    print(num_hypo_hyper_pairs_matches, 'matchess for hyponym-hypernym pairs')

    finded_sentences = []
    with gzip.open(text_corpus, 'rt', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=total_lines)):
            if i in potential_sentences_ids:
                finded_sentences.append(line.strip())
    with open(output_file, 'w') as f:
        for sentence in finded_sentences:
            print(sentence, file=f)


def create_index(text_corpus: str, output_file: str, threshold: int = 10000, total_lines=132498489):
    index = defaultdict(set)
    with gzip.open(text_corpus, 'rt', encoding='utf-8') as f:
        for i, line in tqdm(enumerate(f), total=total_lines):
            tokens = set(line.strip().lower().split())
            for token in tokens:
                if len(index[token]) <= threshold:
                    index[token].add(i)
    for token in tqdm(index, total=len(index), desc='set to list convertation'):
        index[token] = list(index[token])
    json.dump(index, open(output_file, 'w'))


def text_stats(text_corpus):
    counter = defaultdict(int)
    with gzip.open(text_corpus, 'rt', encoding='utf-8') as f:
        for line in f:
            tokens = line.strip().split()
            for token in tokens:
                counter[token] += 1
    print('number of words:', len(counter))
    print('min word frequency:', min(counter.values()))
    print('max word frequency:', max(counter.values()))
    print('mean word frequency:', sum(counter.values()) / len(counter))


def non_overlapping_spans(span, spans):
    start, end = span
    return [(x, y) for (x, y) in spans if not (start <= x <= y <= end)]


def create_tags_sequence(target_spans, sequence_len, neg_tag, pos_tag = 'HYPER'):
    tags_sequence = [neg_tag] * sequence_len
    for span in target_spans:
        for i, pos in enumerate(range(*span)):
            tags_sequence[pos] = ('I-' if i else 'B-') + pos_tag
    return tags_sequence


def create_hyponym_hypernym_dataset(
    hyponyms_file: str, hypernyms_file: str,
    sentences_file: str, output_file: str, neg_tag = 'O',
    max_examples = 20000
):
    examples = []
    num_examples = 0

    hyponyms = [line.strip().lower().split('\t')[0].split() for line in open(hyponyms_file).readlines()]
    hyponym_hypernyms = [
        [hypernym.split() for hypernym in line.strip().lower().split('\t')] for
            line in open(hypernyms_file).readlines()
    ]
    print(hyponyms[0])
    print(hyponym_hypernyms[0])
    sentences = [line.strip().split() for line in open(sentences_file).readlines()]
    lowered_sentences = [line.strip().lower().split() for line in open(sentences_file).readlines()]

    for hyponym_id, (hyponym, hypernyms) in tqdm(
        enumerate(zip(hyponyms, hyponym_hypernyms)),
        total=len(hyponyms),
        desc='creating examples ...'
    ):
        hyponym_len = len(hyponym)
        hyponym_str = ' '.join(hyponym)

        for hypernym_id, hypernym in enumerate(hypernyms):
            needed_tokens = hyponym + hypernym
            # print(needed_tokens)
            hypernym_str = ' '.join(hypernym)
            hypernym_len = len(hypernym)
            filtered_sentences = [
                (i, sentence) for i, sentence in enumerate(lowered_sentences) if all([token in set(sentence) for token in needed_tokens])
            ]
            for (sent_id, sentence) in filtered_sentences:
                hyponym_starts = [
                    i for i in range(len(sentence) - hyponym_len)
                    if sentence[i:i + hyponym_len] == hyponym
                ]
                hyponym_spans = [(i, i + hyponym_len) for i in hyponym_starts]
                if not hyponym_spans:
                    continue
                hypernym_starts = [
                    i for i in range(len(sentence) - hypernym_len)
                    if sentence[i:i + hypernym_len] == hypernym
                ]
                hypernym_spans = [(i, i + hypernym_len) for i in hypernym_starts]

                for hyponym_span_id, hyponym_span in enumerate(hyponym_spans):
                    target_hypernym_spans = non_overlapping_spans(hyponym_span, hypernym_spans)
                    tags_sequence = create_tags_sequence(target_hypernym_spans, len(sentence), neg_tag)
                    examples.append({
                        'id': f'hyponym_id-{hyponym_id}+hypernym_id-{hypernym_id}+sentence_id-{sent_id}+hyponym_span_id-{hyponym_span_id}',
                        'token': sentences[sent_id],
                        'hyponym_span': hyponym_span,
                        'tags_sequence': tags_sequence,
                        'pair': (hyponym_str, hypernym_str)
                    })
                    num_examples += 1

    print('number of dataset examples:', num_examples)
    print(f'saving random {min(max_examples, len(examples))} examples')
    shuffle(examples)
    json.dump(examples[:max_examples], open(output_file, 'w'))


def create_hyponym_dataset(
    hyponyms_file: str,
    sentences_file: str, hyponym_delimeter: str,
    output_file: str
):
    examples = []
    num_examples = 0

    hyponyms = [line.strip().lower().split('\t')[0] for line in open(hyponyms_file).readlines()]
    sentences = [line.strip().lower().split() for line in open(sentences_file).readlines()]

    for hyponym_id, hyponym in tqdm(enumerate(hyponyms), total=len(hyponyms), desc='creating examples ...'):
        hyponym_tokens = hyponym.split()
        hyponym_len = len(hyponym_tokens)

        needed_tokens = hyponym_tokens
        filtered_sentences = [sentence for sentence in sentences if all([token in sentence for token in needed_tokens])]
        for sent_id, sentence in enumerate(filtered_sentences):
            hyponym_starts = [
                i for i in range(len(sentence) - hyponym_len)
                if sentence[i:i + hyponym_len] == hyponym_tokens
            ]
            hyponym_ends = [x + hyponym_len - 1 for x in hyponym_starts]

            if len(hyponym_starts) == 0:
                continue

            new_sentence = []
            for i, token in enumerate(sentence):
                if i in hyponym_starts:
                    new_sentence.append(hyponym_delimeter)

                new_sentence.append(token)

                if i in hyponym_ends:
                    new_sentence.append(hyponym_delimeter)

            examples.append(
                {
                    'id': f'hyponym_id-{hyponym_id}+sentence_id-{sent_id}',
                    'token': new_sentence,
                    'tags_sequence': ['O'] * len(new_sentence)
                }
            )
            num_examples += 1

    print('number of dataset examples:', num_examples)
    json.dump(examples, open(output_file, 'w'))


def data_lookup(source_file: str, bs: int = 5):
    data = json.load(open(source_file))
    shuffle(data)
    idx = 0
    def prepare_print_string(tokens, tags_sequence):
        result = []
        prev_tag = 'O'
        for token, tag in zip(tokens, tags_sequence):
            if tag == 'B-HYPER' and (prev_tag in ['O', 'I-HYPER', 'B-HYPER']):
                result.append('#')
            if (tag == 'B-HYPER' or tag == 'O') and prev_tag in ['I-HYPER', 'B-HYPER']:
                result.append('#')
            result.append(token)
            prev_tag = tag
        return ' '.join(result)

    while True:
        print(*[prepare_print_string(x['token'], x['tags_sequence']) for x in data[idx: idx + bs]], sep='\n\n')
        cmd = input()
        if cmd == 'w':
            idx = max(0, idx - bs)
        elif cmd == 's':
            idx = min(len(data) - bs, idx + bs)
        elif cmd == 'q':
            break


if __name__ == '__main__':
    random.seed(42)
    fire.Fire(create_hyponym_hypernym_dataset)
