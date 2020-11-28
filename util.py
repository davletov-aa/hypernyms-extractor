import os
import fire
from collections import defaultdict
from tqdm import tqdm
from random import randint, shuffle
import json


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



def extract_potential_definition_sentences(
	hyponyms_file: str, hypernyms_file: str, index_file: str,
	text_corpus: str, output_file: str, only_hyponyms: bool = False, max_num_sents: int = 100
):

	hyponyms = [line.lower() for line in open(hyponyms_file).readlines()]
	hypernyms = [line.lower() for line in open(hypernyms_file).readlines()]

	assert len(hyponyms) == len(hypernyms), f"len(hyponyms) = {len(hyponyms)}; len(hypernyms) = {len(hypernyms)}"
	print('loading index file')
	index = json.load(open(index_file))
	for token in tqdm(index, total=len(index), desc='index convertation ...'):
		index[token] = set(index[token])

	potential_sentences_ids = set()
	finded_sentences = []
	num_matches_for_hyponym_hypernym_pairs = 0

	for hyponym, hyponym_hypernyms in tqdm(
		zip(hyponyms, hypernyms),
		total=len(hyponyms),
		desc='extracting potential definition sentences ...'
	):
		hyponym_tokens = hyponym.strip().split('\t')[0].split()
		hyponym_hypernyms = hyponym_hypernyms.strip().split('\t')
		success = True
		if hyponym_tokens[-1] in index:
			hyponym_ids = index[hyponym_tokens[-1]]
		else:
			continue
		for token in hyponym_tokens[:-1]:
			if token in index:
				hyponym_ids = hyponym_ids.intersection(index[token])
			else:
				success = False
				break
		if not success:
			continue

		if only_hyponyms:
			hyponym_ids = list(hyponym_ids)
			shuffle(hyponym_ids)
			hyponym_ids = set(hyponym_ids[:max_num_sents])
			potential_sentences_ids = potential_sentences_ids.union(hyponym_ids)
			continue

		first_finded_sentence = True
		for hypernym in hyponym_hypernyms:
			needed_tokens = hypernym.split()
			cur_search_ids = hyponym_ids
			for token in needed_tokens:
				if token in index:
					cur_search_ids = cur_search_ids.intersection(index[token])
				else:
					success = False
					break
			if not success:
				continue
			if first_finded_sentence:
				first_finded_sentence = False
				num_matches_for_hyponym_hypernym_pairs += 1
			potential_sentences_ids = potential_sentences_ids.union(cur_search_ids)

	print('found', len(potential_sentences_ids), 'potential definition sentences')
	print(num_matches_for_hyponym_hypernym_pairs, 'matchess for hyponym-hypernym pairs')

	with open(text_corpus) as f:
		for i, line in enumerate(tqdm(f, total=132498489)):
			if i in potential_sentences_ids:
				finded_sentences.append(line.strip())
	with open(output_file, 'w') as f:
		for sentence in finded_sentences:
			print(sentence, file=f)


def create_index(text_corpus: str, output_file: str, threshold: int = 500):
	index = defaultdict(set)
	with open(text_corpus) as f:
		for i, line in tqdm(enumerate(f), total=132498489):
			tokens = line.strip().lower().split()
			for token in tokens:
				if len(index[token]) < threshold:
					index[token].add(i)
	for token in index:
		index[token] = list(index[token])
	json.dump(index, open(output_file, 'w'))


def create_hyponym_hypernym_dataset(
	hyponyms_file: str, hypernyms_file: str,
	sentences_file: str, hyponym_delimeter: str,
	output_file: str
):
	examples = []
	num_examples = 0

	hyponyms = [line.strip().lower().split('\t')[0] for line in open(hyponyms_file).readlines()]
	hypernyms = [line.strip().lower().split('\t') for line in open(hypernyms_file).readlines()]
	sentences = [line.strip().lower().split() for line in open(sentences_file).readlines()]

	for hyponym_id, (hyponym, hyponym_hypernyms) in tqdm(enumerate(zip(hyponyms, hypernyms)), total=len(hyponyms), desc='creating examples ...'):
		hyponym_tokens = hyponym.split()
		hyponym_len = len(hyponym_tokens)
		for hypernym_id, hypernym in enumerate(hyponym_hypernyms):
			hypernym_tokens = hypernym.split()
			hypernym_len = len(hypernym_tokens)

			needed_tokens = hyponym_tokens + hypernym_tokens
			filtered_sentences = [sentence for sentence in sentences if all([token in sentence for token in needed_tokens])]
			for sent_id, sentence in enumerate(filtered_sentences):
				hyponym_starts = [
					i for i in range(len(sentence) - hyponym_len)
					if sentence[i:i + hyponym_len] == hyponym_tokens
				]
				hyponym_ends = [x + hyponym_len - 1 for x in hyponym_starts]
				if len(hyponym_starts) == 0:
					continue
				hypernym_starts = [
					i for i in range(len(sentence) - hypernym_len)
					if sentence[i:i + hypernym_len] == hypernym_tokens
				]
				hypernym_ends = [x + hypernym_len - 1 for x in hypernym_starts]
				new_sentence = []
				new_sentence_tags = []
				for i, token in enumerate(sentence):
					if i in hyponym_starts:
						new_sentence.append(hyponym_delimeter)
						new_sentence_tags.append('O')

					new_sentence.append(token)
					is_tag_added = False
					for hypernym_start, hypernym_end in zip(hypernym_starts, hypernym_ends):
						if hypernym_start <= i <= hypernym_end:
							if not is_tag_added:
								new_sentence_tags.append('B-HYPER' if i == hypernym_start else 'I-HYPER')
							is_tag_added = True
					if not is_tag_added:
						new_sentence_tags.append('O')

					if i in hyponym_ends:
						new_sentence.append(hyponym_delimeter)
						new_sentence_tags.append('O')
				examples.append(
					{
						'id': f'hyponym_id-{hyponym_id}+hypernym_id-{hypernym_id}+sentence_id-{sent_id}',
						'token': new_sentence,
						'tags_sequence': new_sentence_tags
					}
				)
				num_examples += 1
	print('number of dataset examples:', num_examples)
	json.dump(examples, open(output_file, 'w'))

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
	fire.Fire(create_hyponym_dataset)
