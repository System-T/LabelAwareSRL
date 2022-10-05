"""
Convert conll2009 data from the '|||'-separated format to token\ttag\n format
"""

for fname in ['conll2009.train.txt','conll2009.valid.txt','conll2009.test.wsj.txt','conll2009.test.brown.txt']:
	with open(fname) as fr, open(fname[:-4]+'.conll','w') as fw:
		for line in fr:
			tokens = line.strip().split(' ||| ')[0].split()[1:]
			labels = line.strip().split(' ||| ')[1].split()
			for token, label in zip(tokens, labels):
				if label == 'B-V':
					fw.write('[SEP]' +'\t' + 'X' + '\n')
				fw.write(token +'\t' + label + '\n')
				if label == 'B-V':
					fw.write('[SEP]' +'\t' + 'X' + '\n')
			fw.write('\n')