# Remove AC examples whose predicate sense does not exist in frames
# Usage: python purge_data_by_predicate path_to_sent_file

import sys
import os
import pickle

infname = sys.argv[1]
outfname = infname[:-5] + '.purged.sent'

try:
  if sys.argv[2] == 'zh':
    lang = 'zh'
except:
  lang = 'en'

if lang == 'en':
  with open('pickles/predicate_role_definition_noun.pkl','rb') as f:
    predicate_role_definition_noun = pickle.load(f)
  with open('pickles/predicate_role_definition_verb.pkl','rb') as f:
    predicate_role_definition_verb = pickle.load(f)

  with open('pickles/predicate_role_definition_noun_.pkl','rb') as f:
    predicate_role_definition_noun_ = pickle.load(f)
  with open('pickles/predicate_role_definition_verb_.pkl','rb') as f:
    predicate_role_definition_verb_ = pickle.load(f)
elif lang == 'zh':
  with open('pickles/zh_predicate_role_definition.pkl','rb') as f:
    predicate_role_definition_noun = pickle.load(f)
  with open('pickles/zh_predicate_role_definition.pkl','rb') as f:
    predicate_role_definition_verb = pickle.load(f)

predicate_role_definition = {'N': predicate_role_definition_noun, 'V': predicate_role_definition_verb}
# from PropBank and NomBank
if lang == 'en':
  predicate_role_definition_reserve = {'N': predicate_role_definition_noun_, 'V': predicate_role_definition_verb_}

with open(infname) as fr, open(outfname,'w') as fw:
  remove_counter = 0
  for line in fr:
    predicate_sense = line.split(' ||| ')[3]
    if predicate_sense == '%.01':
      predicate_sense = "perc-sign.01"
    pos = line.split(' ||| ')[4][0]
    if pos not in ['N','V']:
      pos = 'V'
    if predicate_sense in predicate_role_definition[pos]:
      fw.write(line)
    elif predicate_sense in predicate_role_definition_reserve[pos]:
      fw.write(line)
    else:
      remove_counter += 1
      print(predicate_sense)
    """
    if lang == 'en' and pos == 'N':
      if predicate_sense in predicate_role_definition_noun:
        fw.write(line)
      else:
        remove_counter += 1
        print(predicate_sense)
    elif lang == 'en' and pos == 'V':
      if predicate_sense in predicate_role_definition_verb:
        fw.write(line)
      else:
        remove_counter += 1
        print(predicate_sense)
    else:
      if predicate_sense in predicate_role_definition_noun:
        fw.write(line)
      else:
        remove_counter += 1
        print(predicate_sense)
    """
  print(f'{remove_counter} examples are removed')
