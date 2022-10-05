"""
Train, evaluate and predict with SRL models
"""

from pathlib import Path
import re
import argparse
import pickle
from collections import Counter
from transformers import Trainer, TrainingArguments
from transformers import DistilBertTokenizerFast, DistilBertForTokenClassification, DistilBertForSequenceClassification
from transformers import BertTokenizerFast, BertForTokenClassification, BertForSequenceClassification
from transformers import RobertaTokenizerFast, RobertaForTokenClassification, RobertaForSequenceClassification
from transformers import XLMRobertaTokenizerFast, XLMRobertaForTokenClassification, XLMRobertaForSequenceClassification
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score as calculate_acc
import os
from datasets import load_metric
#from transformers.models.bert.modeling_bert import BertLSTMForTokenClassification
#from transformers.models.roberta.modeling_roberta import RobertaLSTMForTokenClassification
from micro_f1 import sem_f1_score

#import wandb
#wandb.login()

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="Finetune a model on SRL data", action="store_true")
parser.add_argument("--resume", help="If --train, whether to load the last checkpoint.", action="store_true")
parser.add_argument("--eval", help="Evaluate a model on SRL data", action="store_true")
parser.add_argument("--pred", help="Make predictions", action="store_true")
parser.add_argument("--domain", type=str, help="Work with domain-specific datasets", action="store", default='')
parser.add_argument("--domain_labels", help="Expand the label set to include those from domain-specific datasets", action="store_true")
parser.add_argument("--load_model", type=str, help="Load a particular model", action="store", default='')
parser.add_argument("--lstm", help="Put an LSTM on top of transformers", action="store_true")
parser.add_argument("--class_weight", help="Pickle a class weight dictionary", action="store_true")
parser.add_argument("--class_weight_fixed", help="Pickle a class weight dictionary, where all A is 10 times of O", action="store_true")
parser.add_argument("--class_weight_uniform", help="Pickle a class weight dictionary, where all A-tags share the same weight", action="store_true")
parser.add_argument("--focal_loss", help="Use focal loss.", action="store_true")
parser.add_argument('--task', action='store', type=str, help='Subtask: AC or WSD', default='AC')
parser.add_argument('--ac_format', action='store', type=int, help='Format of AC, i.e. the postion of [SEP] in data.', default=1)
parser.add_argument('--purged', action='store_true', help='Whether to use data only with predicates in frames.', default=False)
parser.add_argument('--acd_contextual', action='store_true', help='If --task=ACD, whether to include definitions of contextual arguments too.', default=False)
parser.add_argument('--acd_predsense', action='store_true', help='If --task=ACD, whether to use predicted sense to find definitions.', default=False)
parser.add_argument('--acd_nodef', action='store_true', help='If --task=ACD, whether to use only argument labels but not definitions.', default=False)
parser.add_argument('--acd_bucketcont', action='store_true', help='If --task=ACD, whether to bucket the contextual arugments without definition.', default=True)
parser.add_argument('--transfer', action='store', type=str, help='Load a pretrained model to perform transfer learning.', default='')
parser.add_argument('--train_size_index', action='store', type=str, help='Size of training data and the index of randomization, e.g. 1000_2.', default='')
parser.add_argument('--crosslingual', action='store', type=str, help='Path to training data containing examples in two languages.', default='')
parser.add_argument('--verbsense', action='store_true', help='If --task=WSD, whether to use verb.sense as label.', default=False)
parser.add_argument('--lang', action='store', type=str, help='The language to work with.', default='en')
parser.add_argument('--model', action='store', type=str, help='The model to use.', default='roberta-base')
parser.add_argument('--bsize', action='store', type=int, help='Training batch size.', default=16)
parser.add_argument('--ebsize', action='store', type=int, help='Evaluation batch size.', default=64)
parser.add_argument('--epochs', action='store', type=int, help='Number of epochs', default=10000)

args = parser.parse_args()

# The batch size is the maximum possible for a 16G GPU
if args.model == 'mbert':
  args.model = 'bert-base-multilingual-cased'
  args.bsize = 8
if args.model == 'roberta':
  args.model = 'roberta-base'
  args.bsize = 16
if args.model == 'roberta-large':
  args.bsize = 1

if args.task != 'ACD':
  args.acd_predsense = False

# The weight experiments use environment variables to communicate with
# the HuggingFace model files, which are not included in the repo.
# Hence, these are unavailable for now.
if args.class_weight:
  os.environ["SRL_LOSS"] = "class_weight"
elif args.class_weight_fixed:
  os.environ["SRL_LOSS"] = "class_weight_fixed"
elif args.class_weight_uniform:
  os.environ["SRL_LOSS"] = "class_weight_uniform"
elif args.focal_loss:
  os.environ["SRL_LOSS"] = "focal_loss"
else:
  os.environ["SRL_LOSS"] = "cross_entropy"

if args.train:
    args.eval = True
    args.pred = True

# The LSTM experiments require modified HuggingFace model files, 
# which are not included in the repo.
# Hence, these are unavailable for now.    
if args.lstm:
  model_key = args.model + '-lstm'
else:
  model_key = args.model

code_to_lang = {
    'en': 'English',
    'zh': 'Chinese',
    'ca': 'Catalan',
    'es': 'Spanish',
    'de': 'German',
    'cz': 'Czech',
}

model_to_tokenizer = {
    'bert-base-cased': BertTokenizerFast,
    'bert-large-cased': BertTokenizerFast,
    'bert-large-cased-lstm': BertTokenizerFast,
    'distilbert-base-cased': DistilBertTokenizerFast,
    'distilbert-base-uncased': DistilBertTokenizerFast,
    'distilbert-base-multilingual-cased': DistilBertTokenizerFast,
    'bert-base-multilingual-cased': BertTokenizerFast,
    'roberta-base': RobertaTokenizerFast,
    'roberta-base-lstm': RobertaTokenizerFast,
    'roberta-large': RobertaTokenizerFast,
    'roberta-large-lstm': RobertaTokenizerFast,
    'xlm-roberta-large': XLMRobertaTokenizerFast,
}

model_to_token_classifier = {
    'bert-base-cased': BertForTokenClassification,
    'bert-large-cased': BertForTokenClassification,
    #'bert-large-cased-lstm': BertLSTMForTokenClassification,
    'distilbert-base-cased': DistilBertForTokenClassification,
    'distilbert-base-uncased': DistilBertForTokenClassification,
    'distilbert-base-multilingual-cased': DistilBertForTokenClassification,
    'bert-base-multilingual-cased': BertForTokenClassification,
    'roberta-base': RobertaForTokenClassification,
    #'roberta-base-lstm': RobertaLSTMForTokenClassification,
    'roberta-large': RobertaForTokenClassification,
    #'roberta-large-lstm': RobertaLSTMForTokenClassification,
    'xlm-roberta-large': XLMRobertaForTokenClassification,
}

model_to_sequence_classifier = {
    'bert-base-cased': BertForSequenceClassification,
    'bert-large-cased': BertForSequenceClassification,
    'distilbert-base-cased': DistilBertForSequenceClassification,
    'distilbert-base-uncased': DistilBertForSequenceClassification,
    'distilbert-base-multilingual-cased': DistilBertForSequenceClassification,
    'bert-base-multilingual-cased': BertForSequenceClassification,
    'roberta-base': RobertaForSequenceClassification,
    'roberta-large': RobertaForSequenceClassification,
    'xlm-roberta-large': XLMRobertaForSequenceClassification,
}

metric_f1 = load_metric("f1")
metric_acc = load_metric("accuracy")

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

def read_ac_data(file_path):
    """Get tokens and tags from AC or ACD data."""
    file_path = Path(file_path)
    raw_text = file_path.read_text().strip()
    raw_docs = re.split(r'\n\t?\n', raw_text)
    token_docs = []
    tag_docs = []
    for doc in raw_docs:
        tokens = []
        tags = []
        for line in doc.split('\n'):
            try:
                token, tag = line.split('\t')
            except:
                continue
            tokens.append(token)
            tags.append(tag)
        token_docs.append(tokens)
        tag_docs.append(tags)

    return token_docs, tag_docs

def read_wsd_data(file_path):
    """Get tokens and tags from WSD data."""
    sent_docs = []
    tag_docs = []
    with open(file_path) as fr:
        for line in fr:
            try:
                sent_docs.append(line.strip().split('\t')[0])
                if args.verbsense:
                    tag_docs.append(line.strip().split('\t')[1])
                else:
                    tag_docs.append(line.strip().split('\t')[1][-2:])
            except IndexError:
                continue

    return sent_docs, tag_docs

def encode_tags(tags, encodings, total_length):
    # Enocde [CLS] and non-first subwords as -100
    # Future: consider CNN layer to consider subwords

    # 28 corresponds to R-AM-MNR, an unused tag in the current mapping as of 8.10
    # It is doubly used as unknown, which accomodates for unseen tag in domain-propbanks
    labels = [[tag2id.get(tag,28) for tag in doc] for doc in tags]
    encoded_labels = []
    for doc_labels, doc_encoding in zip(labels, encodings):
        ids = doc_encoding.word_ids()
        doc_enc_labels = [-100] * total_length
        for i, label in enumerate(doc_labels):
            try:
                doc_enc_labels[ids.index(i)] = label
            except ValueError:
                print(f'{i} is not in the ids')
                print('ids is')
                print(ids)
                continue
        encoded_labels.append(doc_enc_labels)

    return encoded_labels

def calculate_f1(labels, preds):
    meaningful_labels = []
    meaningful_preds = []
    for sent_labels, sent_preds in zip(labels, preds):
        meaningful_sent_labels = []
        meaningful_sent_preds = []
        for label, pred in zip(sent_labels, sent_preds):
            if label != -100:# and label != tag2id.get('B-V',-100) and label != tag2id.get('O',-100):
                meaningful_labels.append(label)
                meaningful_preds.append(pred)
                meaningful_sent_labels.append(label)
                meaningful_sent_preds.append(pred)
  
    (P, R, F1, NP, NR, NF1) = sem_f1_score(zip([id2tag[x] for x in meaningful_preds], [id2tag[x] for x in meaningful_labels]))
    #ret = metric_f1.compute(predictions=meaningful_preds, references=meaningful_labels, average='micro')
    print('=====F1======')
    print(NF1)
    ret = {"f1": NF1}
    return ret

def compute_metrics_f1(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=2)
    return calculate_f1(labels, predictions)

def compute_metrics_acc(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    return metric_acc.compute(predictions=predictions, references=labels)

def sent_id_to_tag(sent_labels):
    return [id2tag[prediction] for prediction in sent_labels]


# Read in data
print('Reading data')
has_ood = True
if args.task == 'AC':
    purged_str = '.purged' if args.purged else ''
    if args.train:
      if args.crosslingual:
        train_texts, train_tags = read_ac_data(f'data/conll09/{args.lang}/few_shot/{str(args.crosslingual)}.tsv')
      elif args.train_size_index:
        train_texts, train_tags = read_ac_data(f'data/conll09/{args.lang}/few_shot/ac_{str(args.train_size_index)}.tsv')
      else:
        train_texts, train_tags = read_ac_data(f'data/conll09/{args.lang}/CoNLL2009-ST-{code_to_lang[args.lang]}-train{purged_str}.ac.tsv')
    else:
      train_texts = []
      train_tags = []
    val_texts, val_tags = read_ac_data(f'data/conll09/{args.lang}/CoNLL2009-ST-{code_to_lang[args.lang]}-development{purged_str}.ac.tsv')
    test_texts, test_tags = read_ac_data(f'data/conll09/{args.lang}/CoNLL2009-ST-evaluation-{code_to_lang[args.lang]}{purged_str}.ac.tsv')
    try:
        test_ood_texts, test_ood_tags = read_ac_data(f'data/conll09/{args.lang}/CoNLL2009-ST-evaluation-{code_to_lang[args.lang]}-ood{purged_str}.ac.tsv')
    except FileNotFoundError:
        has_ood = False
        test_ood_texts = []
        test_ood_tags = []
elif args.task == 'ACD':
  purged_str = '.purged' if args.purged else ''
  predsense_str = '.predsense' if args.acd_predsense else ''
  if args.train:
    if args.train_size_index:
      train_texts, train_tags = read_ac_data(f'data/conll09/{args.lang}/few_shot/acd_{str(args.train_size_index)}.tsv')
    else:
      train_texts, train_tags = read_ac_data(f'data/conll09/{args.lang}/CoNLL2009-ST-{code_to_lang[args.lang]}-train{purged_str}.acd.tsv')
  else:
    train_texts = []
    train_tags = []
  val_texts, val_tags = read_ac_data(f'data/conll09/{args.lang}/CoNLL2009-ST-{code_to_lang[args.lang]}-development{purged_str}.acd.tsv')
  test_texts, test_tags = read_ac_data(f'data/conll09/{args.lang}/CoNLL2009-ST-evaluation-{code_to_lang[args.lang]}{purged_str}{predsense_str}.acd.tsv')
  try:
      test_ood_texts, test_ood_tags = read_ac_data(f'data/conll09/{args.lang}/CoNLL2009-ST-evaluation-{code_to_lang[args.lang]}-ood{purged_str}{predsense_str}.acd.tsv')
  except FileNotFoundError:
      has_ood = False
      test_ood_texts = []
      test_ood_tags = []
elif args.task == 'WSD':
    purged_str = '.purged' if args.purged else ''
    if args.train:
      if not args.train_size_index:
        train_texts, train_tags = read_wsd_data(f'data/conll09/{args.lang}/CoNLL2009-ST-{code_to_lang[args.lang]}-train{purged_str}.wsd.tsv')
      else:
        train_texts, train_tags = read_wsd_data(f'data/conll09/{args.lang}/wsd_{str(args.train_size_index)}.tsv')
    else:
      train_texts = []
      train_tags = []
    val_texts, val_tags = read_wsd_data(f'data/conll09/{args.lang}/CoNLL2009-ST-{code_to_lang[args.lang]}-development{purged_str}.wsd.tsv')
    test_texts, test_tags = read_wsd_data(f'data/conll09/{args.lang}/CoNLL2009-ST-evaluation-{code_to_lang[args.lang]}{purged_str}.wsd.tsv')
    try:
        test_ood_texts, test_ood_tags = read_wsd_data(f'data/conll09/{args.lang}/CoNLL2009-ST-evaluation-{code_to_lang[args.lang]}-ood{purged_str}.wsd.tsv')
    except FileNotFoundError:
        has_ood = False
        test_ood_texts = []
        test_ood_tags = []
elif args.task == 'AC05':
    if args.train:
      if not args.train_size_index:
        train_texts, train_tags = read_ac_data(f'data/conll05/conll05_train.ac.tsv')
      else:
        train_texts, train_tags = read_ac_data(f'') #TODO
    val_texts, val_tags = read_ac_data(f'data/conll05/conll05_dev.ac.tsv')
    test_texts, test_tags = read_ac_data(f'data/conll05/conll05_test.ac.tsv')
    has_ood = False
    test_ood_texts = []
    test_ood_tags = []
elif args.task == 'WSD05':
    if args.train:
      if not args.train_size_index:
        train_texts, train_tags = read_wsd_data(f'data/conll05/conll05_train.sense.tsv')
      else:
        train_texts, train_tags = read_wsd_data(f'') # TODO
    val_texts, val_tags = read_wsd_data(f'data/conll05/conll05_dev.sense.tsv')
    test_texts, test_tags = read_wsd_data(f'data/conll05/conll05_test.sense.tsv')
    has_ood = False
    test_ood_texts = []
    test_ood_tags = []
elif args.task == 'AI':
    if args.train:
      if not args.train_size_index:
        train_texts, train_tags = read_ac_data(f'data/conll09/{args.lang}/CoNLL2009-ST-{code_to_lang[args.lang]}-train.ai.format{args.ac_format}.tsv')
      else:
        train_texts, train_tags = read_ac_data(f'data/conll09/{args.lang}/ai_{str(args.train_size_index)}.tsv')
    val_texts, val_tags = read_ac_data(f'data/conll09/{args.lang}/CoNLL2009-ST-{code_to_lang[args.lang]}-development.ai.format{args.ac_format}.tsv')
    test_texts, test_tags = read_ac_data(f'data/conll09/{args.lang}/CoNLL2009-ST-evaluation-{code_to_lang[args.lang]}.ai.format{args.ac_format}.tsv')
    try:
        test_ood_texts, test_ood_tags = read_ac_data(f'data/conll09/{args.lang}/CoNLL2009-ST-evaluation-{code_to_lang[args.lang]}-ood.ai.format{args.ac_format}.tsv')
    except FileNotFoundError:
        has_ood = False
        test_ood_texts = []
        test_ood_tags = []
elif args.task == 'ACS':
    if args.train:
      if not args.train_size_index:
        train_texts, train_tags = read_ac_data(f'data/conll09/{args.lang}/CoNLL2009-ST-{code_to_lang[args.lang]}-train.acsense.format{args.ac_format}.tsv')
      else:
        train_texts, train_tags = read_ac_data(f'data/conll09/{args.lang}/acsense_{str(args.train_size_index)}.tsv')
    val_texts, val_tags = read_ac_data(f'data/conll09/{args.lang}/CoNLL2009-ST-{code_to_lang[args.lang]}-development.acsense.format{args.ac_format}.tsv')
    test_texts, test_tags = read_ac_data(f'data/conll09/{args.lang}/CoNLL2009-ST-evaluation-{code_to_lang[args.lang]}.acsense.format{args.ac_format}.tsv')
    try:
        test_ood_texts, test_ood_tags = read_ac_data(f'data/conll09/{args.lang}/CoNLL2009-ST-evaluation-{code_to_lang[args.lang]}-ood.acsense.format{args.ac_format}.tsv')
    except FileNotFoundError:
        has_ood = False
        test_ood_texts = []
        test_ood_tags = []
else:
    raise ValueError()

# Note: some domain datasets have labels not in conll09

if args.domain == 'bio':
  print('Loading domain data')
  if args.train:
    # WARNING: using Bio test as val, since Bio does not have val. Make sure to NOT
    # use the best model on "val", but instead keep the ones with the same training steps
    if args.task == 'AC':
      train_texts, train_tags = read_ac_data(f'data/domain_propbank/BioProp.train.purged.ac.tsv')
      val_texts, val_tags = read_ac_data(f'data/domain_propbank/BioProp.test.purged.ac.tsv')
    elif args.task == 'ACD':
      train_texts, train_tags = read_ac_data(f'data/domain_propbank/BioProp.train.purged.acd.tsv')
      val_texts, val_tags = read_ac_data(f'data/domain_propbank/BioProp.test.purged.acd.tsv')
    elif args.task == 'WSD':
      train_texts, train_tags = read_wsd_data(f'data/domain_propbank/bio/BioProp.train.purged.wsd.tsv')
      val_texts, val_tags = read_wsd_data(f'data/domain_propbank/bio/BioProp.test.purged.wsd.tsv')
  if args.task == 'AC':
    test_texts, test_tags = read_ac_data(f'data/domain_propbank/BioProp.test.purged.ac.tsv')
  elif args.task == 'ACD':
    test_texts, test_tags = read_ac_data(f'data/domain_propbank/BioProp.test.purged.acd.tsv')
  elif args.task == 'WSD':
    test_texts, test_tags = read_wsd_data(f'data/domain_propbank/bio/BioProp.test.purged.wsd.tsv')
if args.domain == 'finance':
  if args.task == 'AC':
    test_texts, test_tags = read_ac_data(f'data/domain_propbank/finance_proposition_bank.purged.ac.tsv')
  elif args.task == 'ACD':
    test_texts, test_tags = read_ac_data(f'data/domain_propbank/finance_proposition_bank.purged.acd.tsv')
  elif args.task == 'WSD':
    test_texts, test_tags = read_wsd_data(f'data/domain_propbank/finance_proposition_bank.purged.wsd.tsv')
if args.domain == 'contracts':
  if args.task == 'AC':
    test_texts, test_tags = read_ac_data(f'data/domain_propbank/contracts_proposition_bank.purged.ac.tsv')
  elif args.task == 'ACD':
    test_texts, test_tags = read_ac_data(f'data/domain_propbank/contracts_proposition_bank.purged.acd.tsv')
  elif args.task == 'WSD':
    test_texts, test_tags = read_wsd_data(f'data/domain_propbank/contracts_proposition_bank.purged.wsd.tsv')
if args.domain:
  has_ood = False
  test_ood_texts = []
  test_ood_tags = []

texts = train_texts + val_texts + test_texts + test_ood_texts
tags = train_tags + val_tags + test_tags + test_ood_tags

if args.domain_labels:
  if args.task == 'AC':
    tags += read_ac_data(f'data/domain_propbank/BioProp.ac.tsv')[1]
    tags += read_ac_data(f'data/domain_propbank/finance_proposition_bank.ac.tsv')[1]
    tags += read_ac_data(f'data/domain_propbank/contracts_proposition_bank.ac.tsv')[1]
  elif args.task == 'ACD':
    tags += read_wsd_data(f'data/domain_propbank/bio/BioProp.acd.tsv')[1]
    tags += read_wsd_data(f'data/domain_propbank/finance_proposition_bank.acd.tsv')[1]
    tags += read_wsd_data(f'data/domain_propbank/contracts_proposition_bank.acd.tsv')[1]
  elif args.task == 'WSD':
    tags += read_wsd_data(f'data/domain_propbank/bio/BioProp.wsd.tsv')[1]
    tags += read_wsd_data(f'data/domain_propbank/finance_proposition_bank.wsd.tsv')[1]
    tags += read_wsd_data(f'data/domain_propbank/contracts_proposition_bank.wsd.tsv')[1]
  else:
    raise ValueError()

print('Loading tag mappings')
try:
    if args.task[:3] == 'WSD':
        verbsense_str = 'verbsense' if args.verbsense else 'sense'
        if not args.domain_labels:
          with open(f'pickles/tag_mapping_{args.lang}_{args.task}_{verbsense_str}.pkl', 'rb') as fr:
              unique_tags, tag2id, id2tag = pickle.load(fr)
        else:
          with open(f'pickles/tag_mapping_{args.lang}_{args.task}_{verbsense_str}_domain.pkl', 'rb') as fr:
              unique_tags, tag2id, id2tag = pickle.load(fr)
    else:
        if args.crosslingual:
          if '_acd_' in args.crosslingual:
            cl_str = 'acd' 
          else:
            cl_str = 'ac' 
          with open(f'pickles/tag_mapping_{args.lang}_en_{cl_str}_{args.task}.pkl', 'rb') as fr:
            unique_tags, tag2id, id2tag = pickle.load(fr)
        elif args.domain_labels:
          with open(f'pickles/tag_mapping_{args.lang}_{args.task}_domain.pkl', 'rb') as fr:
            unique_tags, tag2id, id2tag = pickle.load(fr)
        else:
          with open(f'pickles/tag_mapping_{args.lang}_{args.task}.pkl', 'rb') as fr:
            unique_tags, tag2id, id2tag = pickle.load(fr)
            print(f'Loaded', f'pickles/tag_mapping_{args.lang}_{args.task}.pkl')
except FileNotFoundError as e:
    print(e)
    print('Mapping not found. To create a new mapping, comment out the following line and try again.')
    raise SystemExit(0)
    print("Creating new mappings")
    if args.task[:3] != 'WSD':
        unique_tags = set(tag for doc in tags for tag in doc)
    elif args.task[:3] == 'WSD':
        unique_tags = set(tags)
    tag2id = {tag: id for id, tag in enumerate(unique_tags)}
    id2tag = {id: tag for tag, id in tag2id.items()}
    if args.task[:3] != 'WSD':
        tag2id['X'] = -100
        id2tag[-100] = 'X'
    if args.task[:3] == 'WSD':
        verbsense_str = 'verbsense' if args.verbsense else 'sense'
        if not args.domain_labels:
          with open(f'pickles/tag_mapping_{args.lang}_{args.task}_{verbsense_str}.pkl', 'wb') as fw:
            pickle.dump((unique_tags, tag2id, id2tag),fw)
        else:
          with open(f'pickles/tag_mapping_{args.lang}_{args.task}_{verbsense_str}_domain.pkl', 'wb') as fw:
            pickle.dump((unique_tags, tag2id, id2tag),fw)
    else:
        if args.crosslingual:
          raise FileNotFoundError('Cannot find crosslingual mapping.')
        elif args.domain_labels:
          with open(f'pickles/tag_mapping_{args.lang}_{args.task}_domain.pkl', 'wb') as fw:
              pickle.dump((unique_tags, tag2id, id2tag),fw)
        else:
          with open(f'pickles/tag_mapping_{args.lang}_{args.task}.pkl', 'wb') as fw:
            pickle.dump((unique_tags, tag2id, id2tag),fw)

print(tag2id)

# Tokenize data
print('Tokenizing data')
if 'roberta' in args.model:
    tokenizer = model_to_tokenizer[model_key].from_pretrained(args.model, add_prefix_space=True)
else:
    tokenizer = model_to_tokenizer[model_key].from_pretrained(args.model)

is_split_into_words = False if args.task[:3] == 'WSD' else True
# Encode together to feed to Dataset class
if args.train:
  train_encodings_all = tokenizer(train_texts, is_split_into_words=is_split_into_words, padding=True, truncation=True)
val_encodings_all = tokenizer(val_texts, is_split_into_words=is_split_into_words, padding=True, truncation=True)
test_encodings_all = tokenizer(test_texts, is_split_into_words=is_split_into_words, padding=True, truncation=True)
if has_ood:
    test_ood_encodings_all = tokenizer(test_ood_texts, is_split_into_words=is_split_into_words, padding=True, truncation=True)
if args.task[:3] == 'WSD':
    if args.train:
      train_labels = [tag2id.get(x,-100) for x in train_tags]
    val_labels = [tag2id.get(x,-100) for x in val_tags]
    test_labels = [tag2id.get(x,-100) for x in test_tags]
    if has_ood:
        test_ood_labels = [tag2id.get(x,-100) for x in test_ood_tags]
else:
    # Encode separately to ensure .word_ids() works
    if args.train:
      train_encodings = [tokenizer(x, is_split_into_words=is_split_into_words, padding=True, truncation=True) for x in train_texts]
    val_encodings = [tokenizer(x, is_split_into_words=is_split_into_words, padding=True, truncation=True) for x in val_texts]
    test_encodings = [tokenizer(x, is_split_into_words=is_split_into_words, padding=True, truncation=True) for x in test_texts]
    if has_ood:
        test_ood_encodings = [tokenizer(x, is_split_into_words=is_split_into_words, padding=True, truncation=True) for x in test_ood_texts]

    if args.train:
      train_seq_length = len(train_encodings_all['input_ids'][0])
    val_seq_length = len(val_encodings_all['input_ids'][0])
    test_seq_length = len(test_encodings_all['input_ids'][0])
    if has_ood:
        test_ood_seq_length = len(test_ood_encodings_all['input_ids'][0])

    # Construct labels with subwords
    if args.train:
      train_labels = encode_tags(train_tags, train_encodings, train_seq_length)
    val_labels = encode_tags(val_tags, val_encodings, val_seq_length)
    test_labels = encode_tags(test_tags, test_encodings, test_seq_length)
    if has_ood:
        test_ood_labels = encode_tags(test_ood_tags, test_ood_encodings, test_ood_seq_length)

# Build HuggingFace Dataset objects
print('Building HuggingFace Dataset objects')
if args.train:
  train_dataset = Dataset(train_encodings_all, train_labels)
val_dataset = Dataset(val_encodings_all, val_labels)
test_dataset = Dataset(test_encodings_all, test_labels)
if has_ood:
    test_ood_dataset = Dataset(test_ood_encodings_all, test_ood_labels)

if args.task[:3] != 'WSD':
  save_dir = f'./results/{args.task}_{model_key}_{args.lang}'
else:
  verbsense_str = 'verbsense' if args.verbsense else 'sense'
  save_dir = f'./results/{args.task}_{verbsense_str}_{model_key}_{args.lang}'
if args.lstm:
  save_dir += '_lstm'
if args.train_size_index:
  save_dir += '_' + str(args.train_size_index)
if args.crosslingual:
  save_dir += '_' + str(args.crosslingual)
if args.transfer:
  save_dir += '_transfer'
if args.class_weight:
  save_dir += '_weight'
if args.class_weight_fixed:
  save_dir += '_weightfixed'
if args.class_weight_uniform:
  save_dir += '_weightuniform'
if args.focal_loss:
  save_dir += '_focalloss'
if args.domain:
  save_dir += f'_{args.domain}'
# --predsense does not affect training
#if args.acd_predsense:
#  save_dir += '_predsense'
if args.acd_contextual:
  save_dir += '_contextual'
if args.acd_nodef:
  save_dir += '_nodef'
#if args.acd_bucketcont:
#  save_dir += '_bucketcont'
if args.purged:
  save_dir += '_purged'

model_mapping = model_to_token_classifier if args.task[:3] != 'WSD' else model_to_sequence_classifier
if args.load_model:
  model = model_mapping[model_key].from_pretrained(args.load_model, num_labels=len(unique_tags))
else:
  if args.train and not args.resume:
      model = model_mapping[model_key].from_pretrained(args.model, num_labels=len(unique_tags))
  elif args.resume:
    dirs = os.listdir(save_dir)
    dirs.sort(reverse=True)
    ckpt_dir = '/' + dirs[0]
    print('Loading checkpoint from ' + save_dir+ckpt_dir)
    model = model_mapping[model_key].from_pretrained(save_dir+ckpt_dir, num_labels=len(unique_tags))
  elif args.pred or args.eval:
      dirs = os.listdir(save_dir)
      dirs.sort()
      ckpt_dir = '/' + dirs[0]
      print('Loading checkpoint from ' + save_dir+ckpt_dir)
      model = model_mapping[model_key].from_pretrained(save_dir+ckpt_dir, num_labels=len(unique_tags))
  elif args.transfer != '':
      print('Loading checkpoint from ' + args.transfer)
      model = model_mapping[model_key].from_pretrained(args.transfer, num_labels=len(unique_tags))

metric_for_best_model = 'f1' if args.task[:3] != 'WSD' else 'accuracy'
load_best_model_at_end_ = False if args.domain else True
save_total_limit_ = -1 if args.domain else 3
  
# Train or evaluate model
training_args = TrainingArguments(
    output_dir=save_dir,          # output directory
    num_train_epochs=args.epochs,             # total number of training epochs
    per_device_train_batch_size=args.bsize,  # batch size per device during training
    per_device_eval_batch_size=args.ebsize,   # batch size for evaluation
    warmup_steps=500,                # number of warmup steps for learning rate scheduler
    weight_decay=0.01,               # strength of weight decay
    logging_dir='./logs',            # directory for storing logs
    logging_steps=100,
    save_total_limit=save_total_limit_,
    evaluation_strategy = 'steps',
    eval_steps = 100,
    load_best_model_at_end=load_best_model_at_end_,
    metric_for_best_model=metric_for_best_model,
    skip_memory_metrics=True,
)

if args.task[:3] != 'WSD':
  compute_metrics = compute_metrics_f1
else:
  compute_metrics = compute_metrics_acc

train_dataset_ = train_dataset if args.train else val_dataset

trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be trained
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset_,         # training dataset
    eval_dataset=test_dataset,             # evaluation dataset
    compute_metrics=compute_metrics,
)

if args.train and not args.resume:
    trainer.train()
elif args.train and args.resume:
    trainer.train(resume_from_checkpoint=args.resume)
if args.eval:
    trainer.evaluate()

if args.pred:
    targets = [('in-domain', test_dataset, test_texts, test_labels, test_tags)]
    if has_ood:
      targets.append(('out-domain', test_ood_dataset, test_ood_texts, test_ood_labels, test_ood_tags))
    for things in targets:
      # AC and ACD, etc.
      if args.task[:3] != 'WSD':
        predictions = np.argmax(trainer.predict(things[1]).predictions, axis=2)
        if args.task == 'ACD' and args.acd_contextual:
          save_dir = f'preds/{args.task}_cont_{model_key}'
        elif args.task == 'ACD' and args.acd_nodef:
          save_dir = f'preds/{args.task}_nodef_{model_key}'
        else:
          save_dir = f'preds/{args.task}_{model_key}'
        if args.domain:
          save_dir += f'_{args.domain}'
        else:
          save_dir += f'_{args.lang}_{things[0]}'
        if args.train_size_index:
          save_dir += f'_{args.train_size_index}'
        if args.crosslingual:
          save_dir += f'_{args.crosslingual}'
        if args.acd_predsense:
          save_dir += f'_predsense'
        if args.purged:
          save_dir += f'_purged'
        save_dir += f'_output.tsv'
        with open(save_dir, 'w') as fw:
            for sent_text, sent_pred, sent_label in zip(things[2], predictions, things[3]):
                token_index = 0
                sent_meaningful_pred = []
                sent_meaningful_label = []
                for token_pred, token_label in zip(sent_pred, sent_label):
                    if token_label != -100:
                        sent_meaningful_pred.append(token_pred)
                        sent_meaningful_label.append(token_label)
                #out_tokens = [x for x in sent_text if x != '[SEP]']
                out_tokens = [x for x in sent_text]
                out_preds = sent_id_to_tag(sent_meaningful_pred)
                out_labels = sent_id_to_tag(sent_meaningful_label)

                i = 0
                sep_counter = 0
                force_write = False
                for o_t in out_tokens:
                  if sep_counter == 3:
                    force_write = True
                #for o_t, o_p, o_l in zip(out_tokens, out_preds, out_labels):
                  if o_t == '[SEP]' or force_write:
                    sep_counter += 1
                    fw.write(o_t + '\t' + 'X' + '\t' + 'X' + '\n')
                    continue
                  try:
                    o_p = out_preds[i]
                    o_l = out_labels[i]
                  except IndexError:
                    continue
                  fw.write(o_t + '\t' + o_l + '\t' + o_p + '\n')
                  i += 1
                fw.write('\n')
      elif args.task[:3] == 'WSD':
        predictions = trainer.predict(things[1]).predictions
        predictions = np.argmax(predictions, axis=1).tolist()
        if args.domain:
          save_dir = f'preds/{args.task}_{verbsense_str}_{model_key}_{args.domain}'
        else:
          save_dir = f'preds/{args.task}_{verbsense_str}_{model_key}_{args.lang}_{things[0]}'
        if args.purged:
          save_dir += f'_purged'
        save_dir += f'_output.tsv'
        with open(save_dir, 'w') as fw:
          for sent_text, sent_pred, sent_label in zip(things[2], predictions, things[4]):
            verbsense_str = 'verbsense' if args.verbsense else 'sense'
            fw.write(sent_text + '\t' + sent_label + '\t' + id2tag[sent_pred] + '\n')

