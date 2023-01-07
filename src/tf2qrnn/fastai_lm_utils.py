"""
Various ULMFit / FastAI related utils
"""
import os

from fastai.text.data import TensorText
from fastcore.foundation import L

from .commons import count_lines_in_text_file


def lr_or_default(lr, learner_obj):
    if lr is not None:
        return lr
    else:
        print("Running the LR finder...")
        lr_min, lr_steep = learner_obj.lr_find(suggestions=True)
        print(f"LR finder results: min rate {lr_min}, rate at steepest gradient: {lr_steep}")
        return lr_min

def get_fastai_tensors(args):
    """ Read pretokenized and numericalized corpora and return them as TensorText objects understood by
        the scantily documented FastAI's voodoo language model loaders.
    """
    L_tensors_train = L()
    L_tensors_valid = L()
    train_ids_list = []
    valid_ids_list = []
    data_sources = [(args['pretokenized_train'], 'trainset', L_tensors_train, train_ids_list)]
    if args.get('pretokenized_valid') is not None:
        data_sources.append((args['pretokenized_valid'], 'validset', L_tensors_valid, valid_ids_list))

    for datasource_path, datasource_name, L_tensors, ids_list in data_sources:
        with open(datasource_path, 'r', encoding='utf-8') as f:
            print(f"Reading {datasource_name} from {datasource_path}")
            num_sents = count_lines_in_text_file(datasource_path)
            cnt = 0
            for line in f:
                if cnt % 10000 == 0: print(f"Processing {datasource_name}: line {cnt} / {num_sents}...")
                tokens = list(map(int, line.split()))
                if args.get('also_return_ids_as_lists') and len(tokens) > args['min_seq_len']:
                    ids_list.append(tokens)
                tokens = TensorText(tokens)
                if len(tokens) > args['min_seq_len']: L_tensors.append(tokens)
                cnt += 1
    if args.get('also_return_ids_as_lists'): # what a beautiful anti-pattern
        return L_tensors_train, L_tensors_valid, train_ids_list, valid_ids_list
    else:
        return L_tensors_train, L_tensors_valid

