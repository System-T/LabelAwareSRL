import sys

sent_file = sys.argv[1]
pred_file = sys.argv[2]
out_file = sys.argv[3]

is_verbsense = sys.argv[4]

with open(sent_file) as f1, open(pred_file) as f2, open(out_file,'w') as f3:
  for l1, l2 in zip(f1,f2):
    row = l1.split(' ||| ')
    if is_verbsense == 'yes':
      row[3] = l2.split('\t')[1]
    elif is_verbsense == 'no':
      row[3] = row[3][:-2] + l2.split('\t')[2]
    row = [x.strip() for x in row]
    f3.write(' ||| '.join(row))
    f3.write('\n')