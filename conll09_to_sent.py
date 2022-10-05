from conll_format_utils import Reader
import os
import sys

def convert_sen(conll09_fn, data_format):
    data = Reader(conll09_fn, data_format)
    sen_fn = conll09_fn+".sent"
    f = open(sen_fn, "w")
    for sen in data.all_sen:
        for pred_id, pred in enumerate(sen.predicates):
            txt = sen.txt
            labels =  [x if x!="_" else "O" for x in sen.arguments[pred_id]  ]
            #print(pred)
            f.write(txt+" ||| "+ str(int(pred[0])-1) + " ||| "+ " ".join(labels) + " ||| "+ pred[2] + " ||| "+ pred[3] + "\n")
    f.close()
    return sen_fn

try:
  target = sys.argv[1]
except:
  target = 'conll09'

if target == 'conll09':
  for lang in ['en', 'zh', 'es', 'de', 'ca', 'cz']:
      for fname in os.listdir(f'data/conll09/{lang}'):
          if fname[-4:] == '.txt':
              conll09_fn = f'/dccstor/zharry1/SRL_PNMA/huggingface/data/conll09/{lang}/{fname}'
              print(conll09_fn)
              convert_sen(conll09_fn, "conll09")
elif target == 'domain':
  for domain in ['bio', 'finance', 'contracts']:
    for fname in os.listdir(f'data/domain_propbank/{domain}'):
      if fname[-7:] == '.conllu':
        print(f'data/domain_propbank/{domain}/{fname}')
        convert_sen(f'data/domain_propbank/{domain}/{fname}', "conllu")
