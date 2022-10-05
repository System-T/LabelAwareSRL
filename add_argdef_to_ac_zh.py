import pickle
import os
import sys

with open('pickles/zh_predicate_role_definition.pkl','rb') as f:
  predicate_role_definition= pickle.load(f)

sent_paths = ['data/conll09/zh']

include_cont = False
try:
  if sys.argv[2] == 'cont':
    include_cont = True
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
    if fname.endswith('.sent'):
      extension = '.acdef' if not include_cont else '.acdefcont'
      with open(os.path.join(sent_path,fname)) as f, open(os.path.join(sent_path,fname)[:-5]+f'{extension}.tsv', 'w') as fw:
        for line in f:
          row = line.strip().split(' ||| ')
          try:
            pos = row[4][0]
            predicate_sense = row[3]
          except IndexError:
            print('No POS or sense')
            print(line)
          try:
            pred_args = predicate_role_definition[predicate_sense]
          except KeyError:
            print(f'{predicate_sense}, {pos} does not exist in frames')
            pred_args = []
          pred_args_dict = {x[0]:x[1] for x in pred_args}
          # Add contextual args
          if include_cont:
            pred_args_dict.update(contextual_arg_to_definition)
          tokens = line.strip().split(' ||| ')[0].split()
          pred_index = int(line.strip().split(' ||| ')[1])
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
