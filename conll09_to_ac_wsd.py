"""
Convert conll2009 data from the '|||'-separated format to token\ttag\n format
"""
import os
import sys

if sys.argv[1] == 'all':
  langs = ['en', 'zh', 'es', 'de', 'ca', 'cz']
  data_path = 'data/conll09/'
elif sys.argv[1] == 'en':
  langs = ['en']
  data_path = 'data/conll09/'
elif sys.argv[1] == 'domain':
  langs = ['domain_propbank']
  data_path = 'data/'
else:
  langs = [sys.argv[1]]
  data_path = 'data/conll09/'

def convert_ac(input_format=1):
  for lang in langs:
    for fname in os.listdir(f'{data_path}/{lang}'):
      if fname[-5:] == '.sent':
        print('.'.join(f'{data_path}/{lang}/{fname}'.split('.')[:-1])+f'.ac.tsv')
        with open(f'{data_path}/{lang}/{fname}') as fr, open('.'.join(f'{data_path}/{lang}/{fname}'.split('.')[:-1])+f'.ac.tsv','w') as fw:
          for line in fr:
            tokens = line.strip().split(' ||| ')[0].split()
            pred_index = int(line.strip().split(' ||| ')[1])
            labels = line.strip().split(' ||| ')[2].split()
            if input_format == 1:
              for i, (token, label) in enumerate(zip(tokens, labels)):
                if i == pred_index:
                  fw.write('[SEP]' +'\t' + 'X' + '\n')
                fw.write(token +'\t' + label + '\n')
                if i == pred_index:
                  fw.write('[SEP]' +'\t' + 'X' + '\n')
              fw.write('\n')
            elif input_format == 2:
              pred = tokens[pred_index]
              for token, label in zip(tokens, labels):
                fw.write(token +'\t' + label + '\n')
              fw.write('[SEP]' +'\t' + 'X' + '\n')
              fw.write(pred +'\t' + 'X' + '\n')
              fw.write('[SEP]' +'\t' + 'X' + '\n')
              fw.write('\n')

def convert_wsd():
  for lang in langs:
    for fname in os.listdir(f'{data_path}/{lang}'):
      if fname[-5:] == '.sent':
        print('.'.join(f'{data_path}/{lang}/{fname}'.split('.')[:-1])+f'.wsd.tsv')
        with open(f'{data_path}/{lang}/{fname}') as fr, open('.'.join(f'{data_path}/{lang}/{fname}'.split('.')[:-1])+f'.wsd.tsv','w') as fw:
          for line in fr:
            tokens = line.strip().split(' ||| ')[0].split()
            pred_index = int(line.strip().split(' ||| ')[1])
            sense = line.strip().split(' ||| ')[3]
            write_tokens = []
            for i, token in enumerate(tokens):
              if i == pred_index:
                write_tokens.append('[SEP]')
              write_tokens.append(token)
              if i == pred_index:
                write_tokens.append('[SEP]')
            fw.write(' '.join(write_tokens))
            fw.write('\t' + sense + '\n')

def convert_ai(input_format=1):
  for lang in langs:
    for fname in os.listdir(f'{data_path}/{lang}'):
      if fname[-5:] == '.sent':
        with open(f'{data_path}/{lang}/{fname}') as fr, open(f'{data_path}/{lang}/{fname}'[:-9]+f'.ai.format{input_format}.tsv','w') as fw:
          for line in fr:
            tokens = line.strip().split(' ||| ')[0].split()
            pred_index = int(line.strip().split(' ||| ')[1])
            labels = line.strip().split(' ||| ')[2].split()
            if input_format == 1:
              for i, (token, label) in enumerate(zip(tokens, labels)):
                if label != 'O':
                  label = 'A'
                if i == pred_index:
                  fw.write('[SEP]' +'\t' + 'X' + '\n')
                fw.write(token +'\t' + label + '\n')
                if i == pred_index:
                  fw.write('[SEP]' +'\t' + 'X' + '\n')
              fw.write('\n')
            elif input_format == 2:
              pred = tokens[pred_index]
              for token, label in zip(tokens, labels):
                if label != 'O':
                  label = 'A'
                fw.write(token +'\t' + label + '\n')
              fw.write('[SEP]' +'\t' + 'X' + '\n')
              fw.write(pred +'\t' + 'X' + '\n')
              fw.write('[SEP]' +'\t' + 'X' + '\n')
              fw.write('\n')

try:
  if sys.argv[2] == 'ac':
    convert_ac(input_format=int(sys.argv[3]))
  elif sys.argv[2] == 'wsd':
    convert_wsd()
  elif sys.argv[2] == 'ai':
    convert_ai(input_format=int(sys.argv[3]))
except IndexError:
  convert_ac()
  convert_wsd()