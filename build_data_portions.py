import random
random.seed(29)
from pathlib import Path

Path("data/conll09/en/few_shot").mkdir(parents=True, exist_ok=True)

portions = [10, 316, 100, 1000, 3162, 10000, 31622, 100000]
seeds = list(range(len(portions)))

with open('data/conll09/en/CoNLL2009-ST-English-train.acd.tsv') as fr:
  sent_blobs = []
  current_metablob = []
  last_predicate = ''
  for blob in fr.read().split('\n\n'):
    sents = blob.split('\n')
    try:
      predicate = sents[sents.index('[SEP]\tX') + 1].split('\t')[0]
    except ValueError:
      continue
    if predicate != last_predicate:
      sent_blobs.append('\n\n'.join(current_metablob))
      current_metablob = []
    current_metablob.append(blob)
    last_predicate = predicate

  for seed in seeds:
    random.seed(seed)
    random.shuffle(sent_blobs)
    for portion in portions:
      with open(f'data/conll09/en/few_shot/acd_{str(portion)}_{seed}.tsv', 'w') as fw:
        fw.write('\n\n'.join(sent_blobs[:portion]))

with open('data/conll09/en/CoNLL2009-ST-English-train.ac.tsv') as fr:
  sent_blobs = fr.read().split('\n\n')
  for seed in seeds:
    random.shuffle(sent_blobs)
    for portion in portions:
      with open(f'data/conll09/en/few_shot/ac_{str(portion)}_{seed}.tsv', 'w') as fw:
        fw.write('\n\n'.join(sent_blobs[:portion]))