"""
Processes ACD output into AC format.
"""

import sys
import argparse

def is_contexutal(arg, lang):
  """Ad-hoc, language-dependent method to check if an argument is contextual."""
  if lang == 'zh':
    return len(arg) in [3,5]
  else:
    return 'AM-' in arg

parser = argparse.ArgumentParser()

parser.add_argument("--target", type=str, action="store", default='', required=True)
parser.add_argument("--train_size", type=str, action="store", default='-1')
parser.add_argument("--purged", action="store_true")
parser.add_argument("--predsense", action="store_true")
# --bucketcont is by default true. To explore otherwise, disable it.
parser.add_argument("--bucketcont", action="store_true", default=True)
# This is for a control experiment of not using definitions.
parser.add_argument("--nodef", action="store_true")

args = parser.parse_args()

purged_str = '_purged' if args.purged else ''
purged_str2 = '.purged' if args.purged else ''
predsense_str = '_predsense' if args.predsense else ''
predsense_str2 = '.predsense' if args.predsense else ''
#bucket_str = '_bucketcont' if args.bucketcont else ''
nodef_str = '_nodef' if args.nodef else ''

if args.target == 'in':
  if args.train_size == '-1':
    acd_pred_path = f'preds/ACD_roberta-base_en_in-domain{predsense_str}{purged_str}_output.tsv'
    ac_pred_path = f'preds/AC_roberta-base_en_in-domain{purged_str}_output.tsv'
    out_path = f'preds/ACD_AC_roberta-base_en_in-domain{predsense_str}{purged_str}_output.tsv'
  else:
    acd_pred_path = f'preds/ACD_roberta-base_en_in-domain_{args.train_size}_output.tsv'
    ac_pred_path = f'preds/AC_roberta-base_en_in-domain_{args.train_size}_output.tsv'
    out_path = f'preds/ACD_AC_roberta-base_en_in-domain_{args.train_size}_output.tsv'
elif args.target == 'out':
  if args.train_size == '-1':
    acd_pred_path = f'preds/ACD_roberta-base_en_out-domain{predsense_str}{purged_str}_output.tsv'
    ac_pred_path = f'preds/AC_roberta-base_en_out-domain{purged_str}_output.tsv'
    out_path = f'preds/ACD_AC_roberta-base_en_out-domain{predsense_str}{purged_str}_output.tsv'
  else:
    acd_pred_path = f'preds/ACD_roberta-base_en_out-domain_{args.train_size}_output.tsv'
    ac_pred_path = f'preds/AC_roberta-base_en_out-domain_{args.train_size}_output.tsv'
    out_path = f'preds/ACD_AC_roberta-base_en_out-domain_{args.train_size}_output.tsv'
elif args.target == 'bio':
  acd_pred_path = f'preds/ACD_roberta-base_bio{purged_str}_output.tsv'
  ac_pred_path = f'preds/AC_roberta-base_bio{purged_str}_output.tsv'
  out_path = f'preds/ACD_AC_roberta-base_bio{purged_str}_output.tsv'
elif args.target == 'finance':
  acd_pred_path = f'preds/ACD_roberta-base_finance{purged_str}_output.tsv'
  ac_pred_path = f'preds/AC_roberta-base_finance{purged_str}_output.tsv'
  out_path = f'preds/ACD_AC_roberta-base_finance{purged_str}_output.tsv'
elif args.target == 'contracts':
  acd_pred_path = f'preds/ACD_roberta-base_contracts{purged_str}_output.tsv'
  ac_pred_path = f'preds/AC_roberta-base_contracts{purged_str}_output.tsv'
  out_path = f'preds/ACD_AC_roberta-base_contracts{purged_str}_output.tsv'
elif args.target == 'zh':
  acd_pred_path = f'preds/ACD_bert-base-multilingual-cased_zh_in-domain{purged_str}{bucket_str}_output.tsv'
  ac_pred_path = f'preds/AC_bert-base-multilingual-cased_zh_in-domain{purged_str}_output.tsv'
  out_path = f'preds/ACD_AC_bert-base-multilingual-cased_zh_in-domain{purged_str}_{is_bucketcont}_output.tsv'

with open(acd_pred_path) as f:
  acd_pred_blobs = f.read().split('\n\n')[:-1]

with open(ac_pred_path) as f:
  ac_pred_blobs = f.read().split('\n\n')[:-1]

acd_pred_lists = [blob.split('\n') for blob in acd_pred_blobs]
ac_pred_lists = [blob.split('\n') for blob in ac_pred_blobs]

combined_pred_lists = []

# Each AC example correspond to N+1 ACD examples, where N is the 
# number of core arguments (+1 comes from the contextual args).
# Hence, keep a pointer to traverse AC and ACD blocks.
# Blocks are separated by \n\n
last_acd_text = []
acd_index = -1
ac_index = -1
current_pred_list = []
while acd_index < len(acd_pred_lists) - 1:
  # Advance ACD pointer
  acd_index += 1
  acd_pred_list = acd_pred_lists[acd_index]
  # Keep everything before the 3rd [SEP], after which is the definition and arg label, as text
  # Such text is the unique identifier of a block
  acd_text = acd_pred_list[:[i for i, x in enumerate(acd_pred_list) if x == '[SEP]\tX\tX'][2]]
  acd_text = [x.split('\t')[0] for x in acd_text]
  acd_predicate = acd_pred_list[acd_pred_list.index('[SEP]\tX\tX')+1].split('\t')[0]
  # Going to a new block
  if acd_text != last_acd_text:
    if last_acd_text:
      combined_pred_lists.append(current_pred_list)
    last_acd_text = acd_text.copy()
    # Advance AC pointer
    ac_index += 1
    ac_pred_list = ac_pred_lists[ac_index]
    ac_predicate = ac_pred_list[ac_pred_list.index('[SEP]\tX\tX')+1].split('\t')[0]
    ac_first_word = ac_pred_list[0].split('\t')[0]
    #print(acd_predicate, ac_predicate)
    # If AC predicate and ACD predicate are different, common cause is that either of them
    # has missing predictions. This could be caused by running inference on different set
    # of data (purged vs. unpurged).
    assert(acd_predicate == ac_predicate)
    current_pred_list = ac_pred_list.copy()
    # Wipes the predictions from AC, because all needed is the format.
    for i, (ac_sent) in enumerate(ac_pred_list):
      tup = current_pred_list[i].split('\t')
      # With --bucketcont default, this should always be True
      if not is_contexutal(tup[2],args.target) or args.bucketcont:
        current_pred_list[i] = '\t'.join((tup[0], tup[1], 'O'))
  target_arg = acd_pred_list[-2].split('\t')[0]
  is_contexutal_block = acd_pred_list[-4].split('\t')[0] == 'contextual'
  for i, (acd_sent, ac_sent) in enumerate(zip(acd_pred_list, ac_pred_list)):
    tup = current_pred_list[i].split('\t')
    # With --bucketcont default, this should always be False
    if not args.bucketcont and is_contexutal(tup[2], args.target):
      continue
    # If token is [SEP], the label is always 'X'
    if current_pred_list[i].split('\t')[0] == '[SEP]':
      current_pred_list[i] = '\t'.join((tup[0], tup[1], 'X'))
    # If this is the first time a token is assigned a label, go ahead.
    # Otherwise (e.g. a token has already been assigned as A1, but now we work with A2),
    # skip.
    elif acd_sent.split('\t')[2] != 'O' and current_pred_list[i].split('\t')[2] == 'O': 
      if args.bucketcont and is_contexutal_block:
        new_arg = acd_sent.split('\t')[2]
      else:
        new_arg = acd_sent.split('\t')[2] + target_arg[-1]
      current_pred_list[i] = '\t'.join((tup[0], tup[1], new_arg))
combined_pred_lists.append(current_pred_list)

with open(out_path, 'w') as fw:
  fw.write('\n\n'.join(['\n'.join(x) for x in combined_pred_lists]))