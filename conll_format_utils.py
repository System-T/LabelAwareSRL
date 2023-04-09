"""
This script reads in both conllu and conll2009 datasets


- Usages:

python conll_format_utils.py --source_filename <filename> --data_format <either conllu or conll09>

- Variables:

Source_filename: str; input filename
data_format: str; choose one from conll09, conllu

"""

from collections import OrderedDict
import argparse
import re


def get_conllu_column_names():
    # There are the columns name taken from UD data
    columns = ["ID", "FORM", "LEMMA", "UPOS", "XPOS", "FEATS", "HEAD", "DEPREL", "FLAG", "VERB"]

    # ID: Word index, integer starting at 1 for each new sentence; may be a range for multiword tokens; may be a decimal number for empty nodes (decimal numbers can be lower than 1 but must be greater than 0).
    # FORM: Word form or punctuation symbol.
    # LEMMA: Lemma or stem of word form.
    # UPOS: Universal part-of-speech tag.
    # XPOS: Language-specific part-of-speech tag; underscore if not available.
    # FEATS: List of morphological features from the universal feature inventory or from a defined language-specific extension; underscore if not available.
    # HEAD: Head of the current word, which is either a value of ID or zero (0).
    # DEPREL: Universal dependency relation to the HEAD (root iff HEAD = 0) or a defined language-specific subtype of one.
    # 4	doivent	devoir	VERB	VERB	_	0	root	_	_
    # 14	ait	avoir	VERB	VERB	_	10	ccomp	_	_

    return columns

def get_conll09_column_names():
    #these coulmens names from CoNLL2009 propbank data,
    # Changed POS to UPOS to have consistency among different formats
    columns = ["ID", "FORM", "LEMMA","PLEMMA" ,"UPOS", "PUPOS", "FEATS","PFEATS", "HEAD", "PHEAD","DEPREL", "PDEPREL","FLAG", "VERB"]
    return columns


class Meta_info():
    def __init__(self, meta):
        self.meta = meta
        self.sen_id = ""
        self.sen_txt = ""
        self.process_meta()

    def process_meta(self):
        for meta_line in self.meta:
            if re.match(r'^# [0-9]+ #', meta_line):#len(meta_line.split("#")) >= 3: #
                self.sen_id = int(meta_line.split("#")[1])
                self.sen_txt = "#".join(meta_line.split("#")[2:]).strip()
            elif len(meta_line.split("sentence-text:"))>=2:
                self.sen_txt =  "sentence-text:".join(meta_line.split("sentence-text:")[1:]).strip()
            elif len(meta_line.split("sentence-text:"))==1 and not re.match(r'^# \d # ', meta_line):
                self.sen_txt =  "#".join(meta_line.split("#")[1:]).strip()


class Token():

    def __init__(self, tok_line, col_names):
        self.id = tok_line[col_names.index("ID")]
        self.form = tok_line[col_names.index("FORM")]
        self.lemma = tok_line[col_names.index("LEMMA")]
        self.upos = tok_line[col_names.index("UPOS")]
        self.xpos = tok_line[col_names.index("XPOS")]
        self.feat = tok_line[col_names.index("FEATS")]
        self.head = tok_line[col_names.index("HEAD")]
        self.deprel = tok_line[col_names.index("DEPREL")]
        self.ispred = self.is_pred(tok_line[col_names.index("FLAG")])
        self.sense = self.sense(tok_line[col_names.index("VERB")])

    def is_pred(self, tok):
        if tok == "_":
            return False
        else:
            return True

    def sense(self, tok):
        if not self.is_pred(tok):
            return "_"
        else:
            return tok.split(".")[-1]

    def __str__(self):
        out = "id: {}, form: {}, lemma: {}, upos: {}, xpos: {}, feat: {}, head: {}, deprel: {}, pred: {}".format(
            self.id,
            self.form,
            self.lemma,
            self.upos,
            self.xpos,
            self.feat,
            self.head,
            self.deprel,
            self.ispred
        )
        return out



class SenChunk():

    def __init__(self, sen, col_names=[]):
        self.sen, self.meta = self.get_sen_meta(sen)
        self.if_no_predicate = 0
        self.n_predicate = 0
        self.n_columns = len(sen[0])
        if col_names==[]:
            # print("Assuming CoNLLU columns")
            self.col_names = get_conllu_column_names()
        else:
            self.col_names = col_names
        self.predicates = self.get_prediate(col_names)
        self.arguments = self.get_arguments()
        self.txt = self.get_text()
        self.n_token = len(self.txt.split(" "))
        self.deprel = self.get_deprel()
        self.tokens = self.get_tokens()


    def get_sen_meta(self, sen):
        """
        Seperate out the sentence from meta information
        :param sen: List of tab seperated text
        :return:
        """
        sent = []
        meta = []
        for tok_line in sen:
            if tok_line[0].split(" ")[0] == "#":
                meta.append(" ".join(tok_line))
            else:
                sent.append(tok_line)

        meta_out = Meta_info(meta)
        return sent, meta_out

    def get_prediate(self,col_names):
        """

        :return: a list of predicates in a sentence, where each pred is a tuple (verb_id, verb, verb.sense, pos)
        """
        if len(self.sen[0]) <= len(self.col_names):
            self.if_no_predicate += 1
            # print("no predicate")
            return []
        # ind = 0
        predicates = []
        for tok_line in self.sen:
            # tok_col = tok.strip().split("\t")
            # for feat in tok_col:
            if tok_line[self.col_names.index("FLAG")] == "Y":
                # start, end = get_tok_position(int(tok_col[0])-1, int(tok_col[0])-1, tokens)
                # get_tok_position(tok_col[0]-1, tokens)
                predicates.append((tok_line[0],
                                   tok_line[self.col_names.index("VERB")].split(".")[0],
                                   tok_line[self.col_names.index("VERB")],
                                   tok_line[self.col_names.index("UPOS")]))

                self.n_predicate += 1
        return predicates

    def get_arguments(self):
        args = OrderedDict()
        arg_columns = list(map(list,zip(*self.sen)))[len(self.col_names):]
        # for ii, pred in enumerate(self.predicates):
        #     args[pred] = arg_columns[ii]
        return arg_columns

    def get_text(self):
        tokenized_txt = "".join(list(zip(*self.sen))[1])
        meta_txt = "".join(self.meta.sen_txt.split(" "))
        if tokenized_txt != meta_txt: # to handle unknown Conllu format text token line
            return " ".join(list(zip(*self.sen))[1])
        else:
            return self.meta.sen_txt

    def get_deprel(self):
        deprel = []
        for tok_line in self.sen:
            deprel.append(tok_line[self.col_names.index("DEPREL")])
        return deprel

    def get_tokens(self):
        tokens = []
        for tok_line in self.sen:
            tok = Token(tok_line, self.col_names)
            # print(tok)
            tokens.append(tok)
        return tokens



    # def __repr__(self):
    #     print("text ", self.txt)


class ReadData():

    def __init__(self, filename):
        self.filename = filename
        self.sen_with_no_predicates = 0
        self.col_names = self.get_column_names()
        self.all_sen = self.get_sen()
        self.n_sen = len(self.all_sen)
        self.predicate_count = self.count_predicates()

    def read_file(self):
        with open(self.filename) as f:
            data = f.readlines()
        chek_new_lines = conllu_format_check(data)
        if chek_new_lines == -1:
            print("Fix number of lines at the end. ")
            return None
        else:
            print("Checked number of new lines at the end of the document")
            return data

    def get_sen(self):
        all_sen = []
        sen = []
        tok_lines = self.read_file()
        for tok_line in tok_lines:
            if tok_line == "\n" or tok_line == "---\n":
                if sen != []:
                    chunk = SenChunk(sen, self.col_names)
                    all_sen.append(chunk)
                    self.sen_with_no_predicates += chunk.if_no_predicate
                sen = []
            # elif line[0] == "#":  # remove all meta data
            #     continue
            else:
                ### To ignore non-int token indexes in label transfer code https://github.ibm.com/SystemT-Research/SRL-Test/issues/81

                tok_index = tok_line.strip().split("\t")[0]
                if re.match(r'^\d-\d', tok_index) or re.match(r'^\d\d-\d', tok_index) or re.match(r'^\d\d\d-\d', tok_index):
                    # print('ignored ',tok_index)
                    continue
                sen.append(tok_line.strip().split("\t"))

        if sen != []:
            all_sen.append(SenChunk(sen, self.col_names))
            self.sen_with_no_predicates += SenChunk(sen, self.col_names).if_no_predicate
        return all_sen

    def count_predicates(self):
        count = 0
        for sen in self.all_sen:
            count += len(sen.predicates)
        return count

    def count_pos(self, pos="UPOS"):
        stat = OrderedDict()
        for sen in self.all_sen:
            for tok in sen.tokens:
                if pos == "UPOS":
                    predpos = tok.upos
                elif pos == "XPOS":
                    predpos = tok.xpos
                else:
                    print("POS can either be UPOS or XPOS")
                if predpos in stat:

                    stat[predpos] += 1
                else:
                    stat[predpos] = 1
        return stat

    def get_predicate_stat(self, pos="UPOS"):
        stat = OrderedDict()
        for sen in self.all_sen:
            if sen.n_predicate > 0:
                for pred in sen.predicates:
                    if pos == "UPOS":
                        predpos = sen.tokens[int(pred[0])-1].upos
                    elif pos == "XPOS":
                        predpos = sen.tokens[int(pred[0]) - 1].xpos
                    else:
                        print("POS can either be UPOS or XPOS")
                    if predpos in stat:

                        stat[predpos] += 1
                    else:
                        stat[predpos] = 1
        return stat

    def data_arg_count(self):
        argCont = {}
        for sen in self.all_sen:
            for pred_id, pred in enumerate(sen.predicates):
                if pred[2] not in argCont:
                    argCont[pred[2]] = {}
                for arg in sen.arguments[pred_id]:
                    if arg != "_":
                        if arg in argCont[pred[2]]:
                            argCont[pred[2]][arg] += 1
                        else:
                            argCont[pred[2]][arg] = 1
        return argCont

    def data_verbsense_count(self):
        verbCont = {}
        for sen in self.all_sen:
            for pred_id, pred in enumerate(sen.predicates):
                if pred[1] not in verbCont:
                    verbCont[pred[1]] = {}
                if pred[2] in verbCont[pred[1]]:
                    verbCont[pred[1]][pred[2]] += 1
                else:
                    verbCont[pred[1]][pred[2]] = 1
        return verbCont



    def get_column_names(self):
        return []


def conllu_format_check(data):
    # just checking new lines at the end of conll u file
    end_line = -1
    count_new_lines = 0
    while data[end_line] == "\n":
        count_new_lines += 1
        end_line = end_line -1
    if count_new_lines != 1:
        print("Expecting a new line at the end of the file.")
        print("There are {} new lines".format(count_new_lines))
        return -1
    else:
        return 0

class Reader(ReadData):

    def __init__(self, input_file, data_format):
        self.data_format = data_format
        super(Reader, self).__init__(input_file)


    def get_column_names(self):
        if self.data_format == "conllu":
            return get_conllu_column_names()
        elif self.data_format == "conll09":
            col_names = get_conll09_column_names()
            ind = col_names.index("PUPOS")
            col_names[ind] = "XPOS"
            return col_names
        else:
            print("Nospecified")

def pred_pos_prop(srl_file, pos):
    import pandas as pd
    data_09 = Reader(srl_file, "conllu")
    pred_count = data_09.get_predicate_stat(pos)
    pos_count = data_09.count_pos(pos)
    df_pred = pd.DataFrame(zip(*[pred_count.keys(), pred_count.values()]), columns=["POS", "pred_count"])
    df_pos = pd.DataFrame(zip(*[pos_count.keys(), pos_count.values()]), columns=["POS", "pos_count"])
    df = pd.merge(df_pred, df_pos, on=['POS'], how="inner", suffixes=('_c', '_ce'))
    df["prop"] = 100 * df["pred_count"] / df["pos_count"]
    return df

def read_large_data(filename):
    """
    To get one sentence at a time from a big file
    :param filename: path to the filename
    :return: Yield one sentence at a time
    """
    with open(filename) as f:
        data = f.readlines()

    all_sen = []
    sen = []
    for line in data:
        if line == "\n" or line == "---\n":
            if sen != []:
                yield sen
            sen = []
        else:
            sen.append(line.strip().split("\t"))
    if sen != []:
        yield sen


def write_raw_csv(conllu_file):
    import pandas as pd
    import os
    path_raw_sen = os.path.dirname(conllu_file)+"/"+"sentences.csv"
    # path_raw_sen = os.path.join(conllu_file.split("/")[:-1])+".csv"
    print(path_raw_sen)
    sen_id =[]
    text = []
    ind = 0
    for sen in read_large_data(conllu_file):
        sen_id.append(ind)
        sen = SenChunk(sen)
        ind += 1
        text.append(sen.txt)
    df = pd.DataFrame(zip(sen_id, text), columns=["id", "text"])
    df.to_csv(path_raw_sen,index=False)
    return path_raw_sen



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Statistics of various predicate forms')
    parser.add_argument('--source_filename', type=str, default='/Users/ishan/git/DataAndEvaluation/data/input_data/srl/en/propbank_09/CoNLL2009-ST-English-train.txt',
                        help='Input labeled conll file')
    # parser.add_argument('--output_filename', type=str,
    #                     default='/Users/ishan/git/DataAndEvaluation/data/pipeline_output/srl/en/propbank_09/conll2009-izumo-10/conll2009-izumo-10-predcount.conllu',
    #                     help='Input labeled conll file')
    parser.add_argument('--data_format', type=str, default='conll09',
                        help='specify the data format')
    args = parser.parse_args()


    data = Reader(args.source_filename, args.data_format)