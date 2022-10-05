"""
Calculate accuracy from a WSD prediction
Usage:
python accuracy.py path_to_WSD_pred
"""

from sklearn.metrics import accuracy_score
import pandas as pd
import sys
import csv

if __name__ == "__main__":
  gold = []
  pred = []

  df = pd.read_csv(sys.argv[1], sep='\t', header=None, quoting=csv.QUOTE_NONE)
  df.columns = ['sent', 'gold', 'pred']
  gold = df['gold'].values
  pred = df['pred'].values
  #gold = [x[1] for x in df['gold'].values.astype(str)]
  #pred = df['pred'].values.astype(str)

  print(accuracy_score(gold, pred))