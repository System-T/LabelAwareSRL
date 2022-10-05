import sys
import os
from subprocess import check_output

def run_evaluation_head(goldpropfile, outfile, argument_count=[], print_output=True, syn=False, conditional_eval=False):
    #if _debug: print("Current file location: {}".format(__file__))
    argument_count = {'missing':0, 'spurious':0}
    script_dir = os.path.dirname(__file__)
    args = ['perl', os.path.join(script_dir,'eval09.pl'), '-g', goldpropfile, '-s', outfile, '-q']
    with open(os.devnull, 'w') as devnull:
        output = check_output(args, stderr=devnull)
        output = output.decode('utf-8')
    # Just want to return labeled and unlabeled semantic F1 scores
    lines = output.split('\n')
    lf1_line = [line for line in lines if line.startswith('  Labeled F1')][0]
    labeled_f1 = float(lf1_line.strip().split(' ')[-1])
    # uf1_line = [line for line in lines if line.startswith('  Labeled precision:')][0]
    uf1_line = [line for line in lines if line.startswith('  Unlabeled F1')][0]
    unlabeled_f1 = float(uf1_line.strip().split(' ')[-1])
    if not conditional_eval:
        argument_count['missing'] = 0
        argument_count["spurious"] = 0
    if print_output:
        print(output)
    if not syn:
        arg_P = 100.0 * (int(
            [line for line in lines if line.startswith('  Labeled precision:')][0].split("+")[0].split("(")[-1]))/ ((int(
            [line for line in lines if line.startswith('  Labeled precision:')][0].split("+")[1].split("(")[-1]) + argument_count["spurious"])+0.0000001)
        sens_P = 100.0 * int(
            [line for line in lines if line.startswith('  Labeled precision:')][0].split("+")[1].split(")")[0]) / (int(
            [line for line in lines if line.startswith('  Labeled precision:')][0].split("+")[2].split(")")[0])+0.0000001)
        arg_R = 100.0 * (int(
            [line for line in lines if line.startswith('  Labeled recall:')][0].split("+")[0].split("(")[-1]))/ ((int(
            [line for line in lines if line.startswith('  Labeled recall:')][0].split("+")[1].split("(")[-1]) + argument_count["missing"])+0.0000001)
        sens_R = 100.0 * int(
            [line for line in lines if line.startswith('  Labeled recall:')][0].split("+")[1].split(")")[0]) / (int(
            [line for line in lines if line.startswith('  Labeled recall:')][0].split("+")[2].split(")")[0])+0.0000001)
        arg_f1 = (2*arg_P*arg_R)/(arg_P+arg_R+0.0000001)
        sens_f1 = (2*sens_P*sens_R)/(sens_P+sens_R+0.0000001)
        print(
            "Argument p: {:.2f}, r: {:.2f}, f1: {}\nSense p: {:.2f}, r: {:.2f}, f1: {}".format(arg_P, arg_R, arg_f1, sens_P,sens_R, sens_f1))
        return (arg_P, arg_R, arg_f1), (sens_P, sens_R, sens_f1)
    else:
        uf1_line = [line for line in lines if line.startswith('  Labeled   attachment score:')][0]
        LAS = float(uf1_line.strip().split(' ')[-2])
        uf1_line = [line for line in lines if line.startswith('  Unlabeled attachment score:')][0]
        UAS = float(uf1_line.strip().split(' ')[-2])
        print("LAS: {} UAS: {}".format(LAS,UAS ))
        return LAS, UAS

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
            if pred_i == "[_UNK_]":
                pred_i = 'O'
            if golden_i == "[_UNK_]":
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

    print('\teval accurate:{:.2f} predict:{} golden:{} correct:{} \n\tP:{:.2f} R:{:.2f} F1:{:.2f} NP:{:.2f} NR:{:.2f} '
          'NF1:{:.2f}'.format(num_correct/total*100, predict_args, golden_args, correct_args, P*100, R*100, F1*100,
                              NP*100, NR *100, NF1 * 100))

    return (P, R, F1, NP, NR, NF1)


#import pandas as pd

#file = "/Downloads/output.tsv"
#df = pd.read_csv(file, sep="\t", header=None)
#df.columns = ["tokens", "predicted" ,  "gold" ]
#df = df.fillna("O")
#sem_f1_score(df[["predicted", "gold"]].values)
code_to_lang = {
    'en': 'English',
    'zh': 'Chinese',
    'ca': 'Catalan',
    'es': 'Spanish',
    'de': 'German',
    'cz': 'Czech',
}

pred_path = sys.argv[2]

if '_in-domain_' in pred_path:
  pred_paths = [pred_path, pred_path.replace('_in-domain_','_out-domain_')]
elif '_out-domain_' in pred_path:
  pred_paths = [pred_path.replace('_out-domain_','_in-domain_'), pred_path]
else:
  raise ValueError('Prediction path malformed.')

if sys.argv[1] == 'en':
  run_evaluation_head(f'data/conll09/en/CoNLL2009-ST-evaluation-English.txt', pred_paths[0])
  run_evaluation_head(f'data/conll09/en/CoNLL2009-ST-evaluation-English-ood.txt', pred_paths[1])
elif sys.argv[1] == 'all':
  for lang in ['ca','cz','de','en','es','zh']:
    print(f"========{lang} in-domain========")
    run_evaluation_head(f'data/conll09/{lang}/CoNLL2009-ST-evaluation-{code_to_lang[lang]}.txt', f'preds/bert-base-multilingual-cased_{lang}_in-domain_output.conll09')
    if lang in ['cz','en','de']:
      print(f"========{lang} out-domain========")
      run_evaluation_head(f'data/conll09/{lang}/CoNLL2009-ST-evaluation-{code_to_lang[lang]}-ood.txt', f'preds/bert-base-multilingual-cased_{lang}_out-domain_output.conll09')
      