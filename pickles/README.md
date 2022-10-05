# Processed frame definitions
`predicate_role_definition_noun.pkl` and `predicate_role_definition_verb.pkl` contain processed frame files from CoNLL09 `nb` and `pb`. Each file is a map of list of tuples. Example: {predicate.01: [(A0,def0),(A1,def1)]}

`predicate_role_definition_noun_.pkl` and `predicate_role_definition_verb_.pkl` contain processed frame files from NomBank and PropBank acquired externally. The format is the same as above. 

`tag_mapping_en_AC|ACD|WSD.pkl` is a tuple of unique_tags, tag2id, id2tag used by `run_srl.py`. It provides a 1-to-1 mapping between label ID's and labels themselves. These mappings are specific to each model trained. 