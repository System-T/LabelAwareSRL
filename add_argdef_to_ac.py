import pickle
import os
import sys

with open('pickles/predicate_role_definition_noun.pkl','rb') as f:
  predicate_role_definition_noun = pickle.load(f)
with open('pickles/predicate_role_definition_verb.pkl','rb') as f:
  predicate_role_definition_verb = pickle.load(f)

with open('pickles/predicate_role_definition_noun_.pkl','rb') as f:
  predicate_role_definition_noun_ = pickle.load(f)
with open('pickles/predicate_role_definition_verb_.pkl','rb') as f:
  predicate_role_definition_verb_ = pickle.load(f)

predicate_role_definition = {'N': predicate_role_definition_noun, 'V': predicate_role_definition_verb}
# from PropBank and NomBank
predicate_role_definition_reserve = {'N': predicate_role_definition_noun_, 'V': predicate_role_definition_verb_}

if sys.argv[1] == 'en':
  sent_paths = ['data/conll09/en']
elif sys.argv[1] == 'domain':
  sent_paths = ['data/domain_propbank']

include_cont = False
try:
  if sys.argv[2] == 'cont':
    include_cont = True
except IndexError:
  pass

bucket_cont = True #TODO: hardcode
try:
  if sys.argv[2] == 'bucket_cont':
    bucket_cont = True
except IndexError:
  pass

contextual_arg_to_definition = {
  #'AM-COM': 'with whom',
  'AM-LOC': 'location',
  'AM-DIR': 'direction',
  #'AM-GOL': 'goal',
  'AM-MNR': 'manner',
  'AM-TMP': 'time',
  'AM-EXT': 'extent',
  #'AM-REC': 'reciprocal',
  'AM-PRD': 'secondary',
  #'AM-PRP': 'purpose',
  'AM-CAU': 'cause',
  'AM-DIS': 'discourse',
  'AM-MOD': 'modal',
  'AM-NEG': 'negation',
  #'AM-DSP': 'direct speech',
  'AM-ADV': 'adverb',
  #'AM-ADJ': 'adjective',
  #'AM-LVB': 'light verb',
  #'AM-CXN': 'construction',

  # not in Propbank,
  'AM-PRT': 'particle',
  'AM-TM': 'time',
  'AM-PNC': 'purpose'
}

for sent_path in sent_paths:
  for fname in os.listdir(sent_path):
    num_out_of_frame = 0
    if fname.endswith('.sent'):
    #if fname == 'CoNLL2009-ST-evaluation-English.purged.sent':
      if include_cont:
        extension = '.acdefcont'
      elif bucket_cont:
        extension = '.acd' #TODO: consistency
      else:
        extension = '.acdef'
      count = 0
      with open(os.path.join(sent_path,fname)) as f, open(os.path.join(sent_path,fname)[:-5]+f'{extension}.tsv', 'w') as fw:
        for line in f:
          count += 1
          row = line.strip().split(' ||| ')
          try:
            pos = row[4][0]
            predicate_sense = row[3]
          except IndexError:
            print('No POS or sense')
            print(line)
            print(count)
          if predicate_sense == '%.01':
            predicate_sense = "perc-sign.01"
          if pos not in ['N','V']:
            pos = 'V'
          if predicate_sense in predicate_role_definition[pos]:
            pred_args = predicate_role_definition[pos][predicate_sense]
          elif predicate_sense in predicate_role_definition_reserve[pos]:
            pred_args = predicate_role_definition_reserve[pos][predicate_sense]
          else:
            print(f'{predicate_sense}, {pos} does not exist in frames')
            pred_args = [('A0','unknown'),('A1','unknown'),('A2','unknown'),('A3','unknown')]
            num_out_of_frame += 1
          pred_args_dict = {x[0]:x[1] for x in pred_args}
          # Add contextual args
          if include_cont:
            pred_args_dict.update(contextual_arg_to_definition)
          tokens = line.strip().split(' ||| ')[0].split()
          try:
            pred_index = int(line.strip().split(' ||| ')[1])
          except IndexError:
            continue
          labels = line.strip().split(' ||| ')[2].split()
          
          for target_label, arg_def in pred_args_dict.items():
            for i, (token, label) in enumerate(zip(tokens, labels)):
              target_label_stripped = target_label[2:] if target_label[:2] in ['R-','C-'] else target_label
              label_stripped = label[2:] if label[:2] in ['R-','C-'] else label
              if label_stripped != target_label_stripped:
                label = 'O'
              else:
                if label[:2] == 'R-':
                  label = 'R-A'
                elif label[:2] == 'C-':
                  label = 'C-A'
                else:
                  label = 'A'
              if i == pred_index:
                fw.write('[SEP]' +'\t' + 'X' + '\n')
              fw.write(token +'\t' + label + '\n')
              if i == pred_index:
                fw.write('[SEP]' +'\t' + 'X' + '\n')

            fw.write('[SEP]' +'\t' + 'X' + '\n')
            for def_token in arg_def.split():
              fw.write(def_token +'\t' + 'X' + '\n')

            fw.write('[SEP]' +'\t' + 'X' + '\n')
            fw.write(target_label +'\t' + 'X' + '\n')
            fw.write('[SEP]' +'\t' + 'X' + '\n')
            fw.write('\n')

          if bucket_cont:
            for i, (token, label) in enumerate(zip(tokens, labels)):
              if 'AM-' not in label:
                label = 'O'
              if i == pred_index:
                fw.write('[SEP]' +'\t' + 'X' + '\n')
              fw.write(token +'\t' + label + '\n')
              if i == pred_index:
                fw.write('[SEP]' +'\t' + 'X' + '\n')

            fw.write('[SEP]' +'\t' + 'X' + '\n')
            fw.write('contextual' +'\t' + 'X' + '\n')

            fw.write('[SEP]' +'\t' + 'X' + '\n')
            fw.write('AM' +'\t' + 'X' + '\n')
            fw.write('[SEP]' +'\t' + 'X' + '\n')
            fw.write('\n')

    print(f'{num_out_of_frame} predicates do not have an assosiated frame.')