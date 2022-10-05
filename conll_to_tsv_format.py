"""
Convert conll2009 data from the '|||'-separated format to token\ttag\n format
"""
import os
import sys

def convert_ag():
	for lang in ['en', 'zh', 'es', 'de', 'ca', 'cz']:
		for fname in os.listdir(f'data/conll09/{lang}'):
			if fname[-5:] == '.sent':
				with open(f'data/conll09/{lang}/{fname}') as fr, open(f'data/conll09/{lang}/{fname}'[:-9]+'.tsv','w') as fw:
					for line in fr:
						tokens = line.strip().split(' ||| ')[0].split()
						labels = line.strip().split(' ||| ')[1].split()
						for token, label in zip(tokens, labels):
							if label == 'B-V':
								fw.write('[SEP]' +'\t' + 'X' + '\n')
							fw.write(token +'\t' + label + '\n')
							if label == 'B-V':
								fw.write('[SEP]' +'\t' + 'X' + '\n')
						fw.write('\n')

def convert_wsd():
	for lang in ['en', 'zh', 'es', 'de', 'ca', 'cz']:
		for fname in os.listdir(f'data/conll09/{lang}'):
			if fname[-4:] == '.txt':
				with open(f'data/conll09/{lang}/{fname}') as fr, open(f'data/conll09/{lang}/{fname}'[:-4]+'.sense.tsv','w') as fw:
					#print(f'data/conll09/{lang}/{fname}')
					for sent_block in fr.read().split('\n\n'):
						if not sent_block or sent_block.isspace():
							continue
						predicate_index_sense = {}
						all_tokens = []
						for i, line in enumerate(sent_block.split('\n')):
							token = line.split('\t')[1] 
							all_tokens.append(token)
							is_predicate = line.split('\t')[12]
							if is_predicate == 'Y':
								predicate_index_sense[i] = line.split('\t')[13]
						for predicate_index, sense in predicate_index_sense.items():
							for i, token in enumerate(all_tokens):
								if i != predicate_index:
									fw.write(token + ' ')
								else:
									fw.write('[SEP] ' + token + ' ' + '[SEP] ')
							fw.write('\t' + sense + '\n')


if sys.argv[1] == 'ac':
	convert_ag()
elif sys.argv[1] == 'wsd':
	convert_wsd()
else:
	print('Need argument.')