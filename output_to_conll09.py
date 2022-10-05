import sys

pred_path = sys.argv[2]

if '_in-domain_' in pred_path:
  pred_paths = [pred_path, pred_path.replace('_in-domain_','_out-domain_')]
elif '_out-domain_' in pred_path:
  pred_paths = [pred_path.replace('_out-domain_','_in-domain_'), pred_path]
else:
  raise ValueError('Prediction path malformed.')

if sys.argv[1] == 'all':
  langs = ['en', 'es','de','zh','ca','cz']
elif sys.argv[1] == 'en':
  langs = ['en']
else:
  langs = [sys.argv[1]]

try:
  task = sys.argv[3]
except IndexError:
  task = 'AC'

code_to_lang = {
    'en': 'English',
    'zh': 'Chinese',
    'ca': 'Catalan',
    'es': 'Spanish',
    'de': 'German',
    'cz': 'Czech',
}

for lang in langs:
  print(lang)
  for target, pred_path in zip(['in-domain','out-domain'], pred_paths):
    try:
      with open(pred_path) as f:
        arg_pred_blobs = f.read().split('\n\n')

      with open(f'preds/WSD_sense_roberta-base_{lang}_{target}_output.tsv') as f:
        sense_pred_blobs = f.read().split('\n')
    except FileNotFoundError as e:
      if target != 'out-domain':
        raise FileNotFoundError(e)
      continue

    pred_blob_index = -1

    orig_fname = f'data/conll09/{lang}/CoNLL2009-ST-evaluation-{code_to_lang[lang]}.txt' if target == 'in-domain' else \
    f'data/conll09/{lang}/CoNLL2009-ST-evaluation-{code_to_lang[lang]}-ood.txt'
    try:
      print(orig_fname)
      with open(orig_fname) as f, \
      open('.'.join(pred_path.split('.')[:-1]) + '.conll09', 'w') as fw:
        to_write = []
        sent_blobs = f.read().split('\n\n')
        count = 0
        for sent_blob in sent_blobs:
          count += 1
          blob_write_lines = []
          for line in sent_blob.split('\n'):
            write_line = line.split('\t')[:14]
            blob_write_lines.append(write_line)
          for j,line in enumerate(sent_blob.split('\n')):
            try:
              is_predicate = line.split('\t')[12] == 'Y'
            except:
              continue
            if is_predicate:
              pred_blob_index += 1
              #print(pred_blob_index)
              blob_write_lines[j][13] = blob_write_lines[j][13].split('.')[0] + '.' + sense_pred_blobs[pred_blob_index].split('\t')[2]
              try:
                pred_blob = arg_pred_blobs[pred_blob_index]
              except IndexError:
                print(sent_blob)
                continue
              pred_blob = '\n'.join([l for l in pred_blob.split('\n') if '[SEP]' not in l])
              #print(blob_write_lines)
              #print(pred_blob)
              #print('\n\n\n')
              for i, blob_write_line in enumerate(blob_write_lines):
                try:
                  pred_line = pred_blob.split('\n')[i]
                except IndexError:
                  #print(pred_blob)
                  raise SystemExit()
                if pred_line.split('\t')[2] not in ['O', 'B-V']:
                  blob_write_lines[i] += [pred_line.split('\t')[2]]
                else:
                  blob_write_lines[i] += ['_']
              
          for write_line in blob_write_lines:
            to_write.append('\t'.join(write_line))
            to_write.append('\n')
          to_write.append('\n')
        fw.write(''.join(to_write[:-2]))
    except FileNotFoundError:
      print('XXX')
      continue
