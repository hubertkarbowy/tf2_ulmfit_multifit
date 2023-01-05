import subprocess
import argparse, re
import json, csv
import tensorflow as tf
from tqdm import tqdm
from sklearn.preprocessing import MultiLabelBinarizer

BEGIN_INDEX = 2
END_INDEX = 3

def file_len(fname):
    """ Nothing beats wc -l """
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

# todo: use named params
def pretty_print_tagged_sequences(subword_pieces, labels, intents, limit=5, gold=None):
    for i in range(len(subword_pieces)):
        if intents not in [None, []]:
            print(f"---> Intent/Header/Query/ATNI: {intents[i]}")
        #l = list(zip(subword_pieces[i], labels[i]))
        #for subword, label in l:
        for j in range(len(subword_pieces[i])):
            if gold is None:
                print("{:<15} {:<5}". format(subword_pieces[i][j], labels[i][j]))
            else:
                print("{:<15} {:<5} {:<5}". format(subword_pieces[i][j], labels[i][j], gold[i][j]))
        if i > limit:
            break

def mk_labels(ls_json):
    labels_set = set()
    for document in ls_json:
        all_tagged = document['label']
        for tagged in all_tagged:
            labels_set.update(tagged['labels'])
    label_index = {label:index for index,label in enumerate(sorted(labels_set))}
    # index_label = {index:label for index,label in enumerate(sorted(labels_set))}
    return label_index


def mark_multiple_spans(context, query):
    """ Return spans of all asterisk-separated items of a `query` that occur in `context` """
    atvis = query.split('*')
    spans = []
    truncating_context = context
    last_span_end = 0
    for atvi in atvis:
        found = re.search(re.escape(atvi), truncating_context)
        if not found:
            print(f"WARNING: Cannot find attribute value `{atvi}` in sentence `{context}`")
            continue
        span = found.span()
        # print(f"Found span {span} in remainder `{truncating_context}`")
        span2 = (last_span_end+span[0], last_span_end+span[1])
        spans.append(span2)
        last_span_end = span2[1]
        truncating_context = truncating_context[span[1]:]
    return spans




########### SUBWORD TOKENIZATION AND LABEL ALIGNMENT IN MANY FLAVORS ####################

def tokenize_and_align_labels_single(*,
                                     spmproc_info,
                                     sent,
                                     max_seq_len,
                                     do_padding=False,
                                     rev_label_map=None,
                                     is_multilabel,
                                     onehot_encoder=None,
                                     add_bos,
                                     add_eos):
    """
    Given a sentencepiece object `spmproc` and a whitespace-pretokenized sentence `s`,
    do the following:
         1. perform subword tokenization
         2. align labels to new subwords
         3. encode labels as integers as laid out in `rev_label_map` (optionally)

    `sent` is a list of tuples whose zeroath elements contain the whitespace-separated token
    and first elements contain labels encoded as strings. If `is_multilabel` is set to True,
    the untagged tokens have an empty array of labels (i.e. []) associated with them.
    Otherwise, it is assumed that such tokens are represented by the label 0.
    """
    spmproc = spmproc_info['spmproc']
    if add_bos and spmproc._add_bos:
        raise ValueError("When realigning labels to subwords, please disable `add_bos` in the Sentencepiece object")
    if add_eos and spmproc._add_eos:
        raise ValueError("When realigning labels to subwords, please disable `add_eos` in the Sentencepiece object")
    if onehot_encoder is not None and rev_label_map is None:
        raise ValueError("Please provide a reverse labels map if requesting to one-hot encode/multiencode the labels")
    no_label_symbol = [] if is_multilabel else 'O'
    sentence_tokens = []
    sentence_ids = []
    sentence_labels = []
    for whitespace_token in sent:
        subwords = spmproc.encode_as_pieces(whitespace_token[0])
        sentence_tokens.extend(subwords)
        sentence_ids.extend(spmproc.encode_as_ids(whitespace_token[0]))
        sentence_labels.extend([whitespace_token[1]]*len(subwords))
    if max_seq_len is not None:
        # minus up to 2 tokens for BOS and EOS since the encoder was trained on sentences with these markers
        num_additional_stripped = int(bool(add_bos)) + int(bool(add_eos))
        sentence_tokens = sentence_tokens[:max_seq_len-num_additional_stripped]
        sentence_ids = sentence_ids[:max_seq_len-num_additional_stripped]
        sentence_labels = sentence_labels[:max_seq_len-num_additional_stripped]
    if add_bos:
        sentence_tokens = [spmproc.id_to_piece(spmproc.bos_id())] + sentence_tokens
        sentence_ids = [spmproc.bos_id()] + sentence_ids
        sentence_labels = [no_label_symbol] + sentence_labels
    if add_eos:
        sentence_tokens = sentence_tokens + [spmproc.id_to_piece(spmproc.eos_id())]
        sentence_ids = sentence_ids + [spmproc.eos_id()]
        sentence_labels = sentence_labels + [no_label_symbol]
    if do_padding:
        if max_seq_len is None:
            raise ValueError("Please specify the maximum sequence length if requesting padding.")
        num_pads = max(0, max_seq_len - len(sentence_tokens))
        sentence_tokens = sentence_tokens + [spmproc_info['pad_piece']]*num_pads
        sentence_ids = sentence_ids + [spmproc_info['pad_id']]*num_pads
        sentence_labels = sentence_labels + [no_label_symbol]*num_pads
        assert len(sentence_tokens) == len(sentence_ids) == len(sentence_labels) == max_seq_len

    # optionally encode labels:
    if rev_label_map:
        if is_multilabel:
            encoded_labels = [[rev_label_map[l] for l in labels] for labels in sentence_labels]
            if onehot_encoder is not None:
                encoded_labels = onehot_encoder.fit_transform([l for l in encoded_labels])
        else:
            encoded_labels = [rev_label_map[l] for l in sentence_labels]
            if onehot_encoder is not None:
                encoded_labels = onehot_encoder.fit_transform([[l] for l in encoded_labels])
    else:
        encoded_labels = None
    return sentence_tokens, sentence_ids, sentence_labels, encoded_labels


def tokenize_and_align_labels(*,
                              spmproc,
                              sents,
                              max_seq_len,
                              do_padding,
                              rev_label_map=None,
                              is_multilabel,
                              onehot_encode,
                              add_bos,
                              add_eos):
    """
    Performs Sentencepiece tokenization on an already whitespace-tokenized text
    and aligns labels to subwords
    """

    print(f"Tokenizing and aligning {len(sents)} examples...")
    if max_seq_len is not None:
        print(f"Note: inputs will be truncated to the first {max_seq_len - 2} tokens")
    if onehot_encode and rev_label_map is None:
        raise ValueError("Please provide a reverse labels map if requesting to one-hot encode/multiencode the labels")
    if onehot_encode:
        onehot_encoder = MultiLabelBinarizer(classes=range(len(rev_label_map)))
    else:
        onehot_encoder = None
    tokenized = []
    numericalized = []
    labels = []
    encoded_labels = []
    spmproc_info = {
        'spmproc': spmproc,
        'pad_id': spmproc.pad_id(),
        'pad_piece': spmproc.id_to_piece(spmproc.pad_id())
    }
    for sent in tqdm(sents):
        sentence_tokens, sentence_ids, sentence_labels, sentence_encoded_labels = \
            tokenize_and_align_labels_single(spmproc_info=spmproc_info,
                                             sent=sent,
                                             max_seq_len=max_seq_len,
                                             do_padding=do_padding,
                                             rev_label_map=rev_label_map,
                                             is_multilabel=is_multilabel,
                                             onehot_encoder=onehot_encoder,
                                             add_bos=add_bos,
                                             add_eos=add_eos)
        tokenized.append(sentence_tokens)
        numericalized.append(sentence_ids)
        labels.append(sentence_labels)
        encoded_labels.append(sentence_encoded_labels)
    return tokenized, numericalized, labels, encoded_labels


def subword_tokenize_and_find_label_spans(*, spm_layer, input_tsv,
                                          sep='\t',
                                          iob_segmentation='b_until_first_whitespace_then_i',
                                          add_bos=False,
                                          add_eos=False,
                                          also_add_bos_eos_to_queries=False,
                                          support_multiple_spans=False):
    """ other options:
        - 'b_on_first_subword_then_i'
        - 'bi_on_first_subwords_then_o'
        - None

        If `support_multiple_spans` is set to True, the input sentence may contain more than one
        span to be marked as an entity. In this case multiple values in the second column should
        be separated with asterisks.
    """
    spmproc = spm_layer.spmproc
    tokenized_contexts = []
    tokenized_attributes = []
    tokenized_contexts_pieces = []
    tokenized_attributes_pieces = []
    tokenized_labels = []

    print("Tokenizing and aligning labels...")
    flen = file_len(input_tsv)
    with open(input_tsv, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=sep)
        for row in tqdm(reader, total=flen): # row[0] = context, row[1] = query, row[2] = value in context or 'NULL'
            # pbar.update(row_idx)
            # print(row)
            context = row[0]
            query = row[1]
            try:
                atvi = row[2]
            except IndexError: # test files do not contain the gold column
                atvi = 'NULL'
            curr_context_tokens = []
            curr_context_pieces = []
            curr_attributes_pieces = []
            curr_attributes_tokens = []
            curr_labels = []
            i = 0
            if atvi == 'NULL':
                spans = []
            else:
                if support_multiple_spans:
                    spans = mark_multiple_spans(context, atvi)
                else:
                    found = re.search(re.escape(atvi), context)
                    if not found:
                        print(f"WARNING: Cannot find ATVI `{row[2]}` in title `{row[0]}`")
                        continue
                    spans = [found.span()]

            ##### CONTEXT WITH ATVI #####
            if add_bos:
                curr_context_tokens.extend([BEGIN_INDEX])
                curr_labels.extend(['O'])
            i = 0
            for span in spans: # if spans not in [None, []]:
                # before span:
                entity_beg = span[0]
                entity_end = span[1]
                res = spmproc.tokenize(context[i:entity_beg]).numpy().tolist()
                curr_context_tokens.extend(res)
                curr_labels.extend(['O']*len(res))

                # inside span:
                if iob_segmentation in ['b_on_first_subword_then_i', 'iob']:
                    res = spmproc.tokenize(context[entity_beg:entity_end]).numpy().tolist()
                    curr_context_tokens.extend(res)
                    curr_labels.extend(['B'] + ['I']*(len(res)-1))
                elif iob_segmentation == 'b_until_first_whitespace_then_i':
                    whitespaced = context[entity_beg:entity_end].split()
                    for word_idx, word in enumerate(whitespaced):
                        res = spmproc.tokenize(word).numpy().tolist()
                        curr_context_tokens.extend(res)
                        curr_labels.extend(['B']*len(res) if word_idx == 0 else ['I']*len(res))
                elif iob_segmentation in [None, 'none', 'b_only']:
                    res = spmproc.tokenize(context[entity_beg:entity_end]).numpy().tolist()
                    curr_context_tokens.extend(res)
                    curr_labels.extend(['B']*len(res))
                else:
                    raise ValueError(f"Unsupported IOB segmentation {iob_segmentation}")
                i = entity_end

            # after all the spans:
            res = spmproc.tokenize(context[i:]).numpy().tolist()
            curr_context_tokens.extend(res)
            curr_labels.extend(['O']*len(res))
            if add_eos:
                curr_context_tokens.extend([END_INDEX])
                curr_labels.extend(['O'])
            curr_context_pieces = [t.decode(encoding='utf-8') for t in spmproc.id_to_string(curr_context_tokens).numpy().tolist()]

            ###### ATNI / QUERY #####:

            res = spmproc.tokenize(query).numpy().tolist()
            if also_add_bos_eos_to_queries and add_bos:
                curr_attributes_tokens.extend([BEGIN_INDEX])
            curr_attributes_tokens.extend(res)
            if also_add_bos_eos_to_queries and add_eos:
                curr_attributes_tokens.extend([END_INDEX])
            curr_attributes_pieces = [t.decode(encoding='utf-8') for t in spmproc.id_to_string(curr_attributes_tokens).numpy().tolist()]

            ####### append to the main lists ######
            tokenized_contexts.append(curr_context_tokens)
            tokenized_attributes.append(curr_attributes_tokens)
            tokenized_contexts_pieces.append(curr_context_pieces)
            tokenized_attributes_pieces.append(curr_attributes_pieces)
            tokenized_labels.append(curr_labels)
    # pbar.close()
    return tokenized_contexts, tokenized_contexts_pieces, tokenized_attributes, tokenized_attributes_pieces, tokenized_labels


def label_studio_to_tagged_subwords(*, spm_layer, label_studio_min_json):
    # spm_layer = SPMNumericalizer(name="SPM_layer",
    #                              spm_path=spm_args['spm_model_file'],
    #                              add_bos=spm_args.get('add_bos') or False,
    #                              add_eos=spm_args.get('add_eos') or False,
    #                              lumped_sents_separator="")
    spmproc = spm_layer.spmproc
    ls_json = json.load(open(label_studio_min_json, 'r', encoding='utf-8'))

    # tokenize with offsets
    nl_texts = [document['text'] for document in ls_json]
    spans = [document.get('label') or [] for document in ls_json]
    print(f"First 10 texts:")
    print(nl_texts[0:10])

    # map spans to subwords
    tokenized = []
    tokenized_pieces = []
    tokenized_labels = []
    intents = []
    for doc_id in range(len(nl_texts)):
        # Tensorflow's tokenize_with_offsets is broken with SentencePiece
        #token_offsets = list(zip(begins[i].numpy().tolist(), ends[i].numpy().tolist()))
        #pieces = [t.decode(encoding='utf-8') for t in spmproc.id_to_string(piece_ids[i]).numpy().tolist()]
        curr_tokens = []
        curr_pieces = []
        curr_entities = []
        i = 0
        if spans[doc_id] == []:
            entity_end=0
        for span in spans[doc_id]:
            j = entity_beg = span['start']
            entity_end = span['end']
            label_class = span['labels'][0] # assume labels don't overlap
            
            # tokenize everything before the label span
            res = spmproc.tokenize(nl_texts[doc_id][i:j]).numpy().tolist()
            curr_tokens.extend(res)
            curr_entities.extend(['O']*len(res))
            
            # inside the label span
            res = spmproc.tokenize(nl_texts[doc_id][j:entity_end]).numpy().tolist()
            curr_tokens.extend(res)
            curr_entities.extend([label_class]*len(res))

        # from the last label to EOS
        res = spmproc.tokenize(nl_texts[doc_id][entity_end:]).numpy().tolist()
        curr_tokens.extend(res)
        curr_entities.extend(['O']*len(res))
        curr_pieces = [t.decode(encoding='utf-8') for t in spmproc.id_to_string(curr_tokens).numpy().tolist()]
        tokenized.append(curr_tokens)
        tokenized_pieces.append(curr_pieces)
        tokenized_labels.append(curr_entities)
        if ls_json[doc_id].get('intent') is not None:
            intents.append(ls_json[doc_id].get('intent'))
    return tokenized, tokenized_pieces, tokenized_labels, intents

def main(args):
    from tf2_ulmfit.ulmfit_tf2 import SPMNumericalizer
    spm_args = {'spm_model_file': args['spm_model_file'], 'add_bos': False, 'add_eos': False}
    spm_layer = SPMNumericalizer(name="SPM_layer",
                                 spm_path=spm_args['spm_model_file'],
                                 add_bos=False, # always False here
                                 add_eos=False,
                                 lumped_sents_separator="")
    if args.get('label_studio_min_json'):
        _, token_pieces, token_labels = label_studio_to_tagged_subwords(spm_layer=spm_layer,
                                                                        label_studio_min_json=args['label_studio_min_json'])
    elif args.get('label_tsv_file'):
        _, token_pieces, _, intents, token_labels = subword_tokenize_and_find_label_spans(spm_layer=spm_layer,
                                                                                      input_tsv=args['label_tsv_file'],
                                                                                      add_bos=True,
                                                                                      add_eos=True,
                                                                                      also_add_bos_eos_to_queries=False,
                                                                                      iob_segmentation='b_until_first_whitespace_then_i')
    else:
        raise ValueError("Please provide either --label-studio-min-json or --label-tsv-file, not none, not both")
    pretty_print_tagged_sequences(token_pieces, token_labels, intents)


if __name__ == "__main__":
    argz = argparse.ArgumentParser()
    argz.add_argument('--label-studio-min-json', required=False)
    argz.add_argument('--label-tsv-file', required=False)
    argz.add_argument('--spm-model-file', required=True)
    argz.add_argument('--add-bos', action='store_true', required=False)
    argz.add_argument('--add-eos', action='store_true', required=False)
    args = vars(argz.parse_args())
    main(args)
