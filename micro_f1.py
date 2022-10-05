from sklearn.metrics import f1_score
import pandas as pd
import sys
import csv

if len(sys.argv) > 2 and sys.argv[2] == 'f1':
  f1_only = True
else:
  f1_only = False

def sem_f1_score(predictions, unify_pred = False, predicate_correct=0, predicate_sum=0, gold_predicate_sum=0, psd=False, write_confusion_path=None):
    """
    predictions: tuples of (predict, gold)
    P, R, F1: comnbined score of disambiguation and argument
    N (P, R, F1): Argument only score
    """
    predict_args = 0
    golden_args = 0
    correct_args = 0
    num_correct = 0
    total = 0

    if unify_pred:
        predicate_correct = 0
        predicate_sum = 0
    if psd:
        print("Predicate accuracy: {}/{}={:.2f}".format(predicate_correct, gold_predicate_sum, 100*float(predicate_correct/gold_predicate_sum)))
        NPP = predicate_correct / (predicate_sum + 1e-13)

        NRP = predicate_correct / (gold_predicate_sum + 1e-13)

        NF1P = 2 * NPP * NRP / (NPP + NRP + 1e-13)
        print("Predicate: predicate_correct:{}, predicate_sum:{}, gold_predicate_sum:{}, \n NPP:{:.2f}, NRP:{:.2f}, NF1P:{:.2f}".format(predicate_correct,
                                                                                                              predicate_sum,gold_predicate_sum,
                                                                                            100*NPP, 100*NRP, 100*NF1P))

    gold__ = []
    for (pred_i, golden_i)  in predictions:
        gold__.append(golden_i)
        if unify_pred:
            predicate_sum += 1
            if pred_i == golden_i:
                predicate_correct += 1
        else:
            if golden_i == "[_PAD_]":
                continue
            total += 1
            if pred_i == "UNK":
                pred_i = 'O'
            if golden_i == "UNK":
                golden_i = 'O'
            if pred_i == "B-V":
                pred_i = 'O'
            if golden_i == "B-V":
                golden_i = 'O'
            if pred_i != 'O':
                predict_args += 1
            if golden_i != 'O':
                golden_args += 1
            if golden_i != 'O' and pred_i == golden_i:
                correct_args += 1
            if pred_i == golden_i:
                num_correct += 1
 
    P = (correct_args + predicate_correct) / (predict_args + predicate_sum + 1e-13)

    R = (correct_args + predicate_correct) / (golden_args + gold_predicate_sum + 1e-13)

    NP = correct_args / (predict_args + 1e-13)

    NR = correct_args / (golden_args + 1e-13)
        
    F1 = 2 * P * R / (P + R + 1e-13)

    NF1 = 2 * NP * NR / (NP + NR + 1e-13)

    if not f1_only:
      print('\teval accurate:{:.2f} predict:{} golden:{} correct:{} \n\tP:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} '
          'NF1:{:.2f}'.format(num_correct/total*100, predict_args, golden_args, correct_args, P*100, R*100, F1*100,
                              NP*100, NR *100, NF1 * 100))
    else:
      print(round(NF1 * 100, 1))

    return (P, R, F1, NP, NR, NF1)


if __name__ == "__main__":
  gold = []
  pred = []

  df = pd.read_csv(sys.argv[1], sep='\t', header=None, quoting=csv.QUOTE_NONE)
  df.columns = ['sent', 'gold', 'pred']
  sents = df['sent']
  gold = df['gold'].values
  pred = df['pred'].values

  golds = []
  preds = []

  for s,g,p in zip(df['sent'], df['gold'].values, df['pred'].values):
    if s != '[SEP]':
      golds.append(g)
      preds.append(p)


  #print(gold)
  sem_f1_score(zip(preds, golds))