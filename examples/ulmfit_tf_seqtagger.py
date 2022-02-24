import argparse
import json
import os, random
import readline

import tensorflow as tf

from lm_tokenizers import LMTokenizerFactory
from ulmfit_commons import check_unbounded_training, print_training_info, prepare_keras_callbacks
from ulmfit_tf2 import STLRSchedule, OneCycleScheduler, RaggedSparseCategoricalCrossEntropy, apply_awd_eagerly
from ulmfit_tf2_heads import ulmfit_sequence_tagger
from pretraining_utils.sequence_utils import tokenize_and_align_labels

# DEFAULT_LABEL_MAP = {0: 'O', 1: 'B-N', 2: 'I-N'}


def r_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def read_labels(label_path):
    label_map = json.load(open(label_path, 'r', encoding='utf-8'))
    label_map = {int(k):v for k,v in label_map.items()}
    rev_label_map = {v:k for k,v in label_map.items()}
    return label_map, rev_label_map

# def tokenize_and_align_labels(spmproc, train_jsonl, max_seq_len, no_label_symbol):
#     """
#     Performs Sentencepiece tokenization on an already whitespace-tokenized text
#     and aligns labels to subwords
#     """

#     print(f"Tokenizing and aligning {len(train_jsonl)} examples...")
#     if max_seq_len is not None:
#         print(f"Note: inputs will be truncated to the first {max_seq_len - 2} tokens")
#     tokenized = []
#     numericalized = []
#     labels = []
#     for sent in train_jsonl:
#         sentence_tokens = []
#         sentence_ids = []
#         sentence_labels = []
#         for whitespace_token in sent:
#             subwords = spmproc.encode_as_pieces(whitespace_token[0])
#             sentence_tokens.extend(subwords)
#             sentence_ids.extend(spmproc.encode_as_ids(whitespace_token[0]))
#             sentence_labels.extend([whitespace_token[1]]*len(subwords))
#         if max_seq_len is not None:
#             # minus 2 tokens for BOS and EOS since the encoder was trained on sentences with these markers
#             sentence_tokens = sentence_tokens[:max_seq_len-2]
#             sentence_ids = sentence_ids[:max_seq_len-2]
#             sentence_labels = sentence_labels[:max_seq_len-2]
#         sentence_tokens = [spmproc.id_to_piece(spmproc.bos_id())] + \
#                           sentence_tokens + \
#                           [spmproc.id_to_piece(spmproc.eos_id())]
#         sentence_ids = [spmproc.bos_id()] + sentence_ids + [spmproc.eos_id()]
#         sentence_labels = [no_label_symbol] + sentence_labels + [no_label_symbol]
#         tokenized.append(sentence_tokens)
#         numericalized.append(sentence_ids)
#         labels.append(sentence_labels)
#     return tokenized, numericalized, labels


def interactive_demo(args):
    ############# settings ############
    label_map = read_labels(args['label_map'])
    spm_args = {'spm_model_file': args['spm_model_file'],
                'add_bos': False,
                'add_eos': False,
                'lumped_sents_separator': '[SEP]'}
    layer_config = {'qrnn': args.get('qrnn'),
                    'num_recurrent_layers': args.get('num_recurrent_layers'),
                    'qrn_zoneout': args.get('qrnn_zoneout') or 0.0}
    spmproc = LMTokenizerFactory.get_tokenizer(tokenizer_type='spm_tf_text',
                                               tokenizer_file=args['spm_model_file'],
                                               add_bos=True, add_eos=True)  # bos/eos will need to be added manually
    activation = 'sigmoid' if args.get('multilabel') else 'softmax'
    ulmfit_tagger, hub_object = ulmfit_sequence_tagger(model_type=args['model_type'],
                                                       pretrained_encoder_weights=None,
                                                       spm_model_args=spm_args,
                                                       fixed_seq_len=args.get('fixed_seq_len'),
                                                       num_classes=len(label_map),
                                                       activation=activation,
                                                       layer_config=layer_config)
    ulmfit_tagger.load_weights(args['model_weights_cp']).expect_partial()
    print("Restored weights successfully")
    ulmfit_tagger.summary()
    readline.parse_and_bind('set editing-mode vi')
    while True:
        sent = input("Write a sentence to tag: ")
        # Our SPMNumericalizer already outputs a RaggedTensor, but in the line below we access
        # the underlying object directly on purpose, so we have to convert it from regular to ragged tensor ourselves.
        subword_ids_tensor = spmproc(tf.constant([sent]))
        subword_ids = subword_ids_tensor.numpy()[0].tolist()
        subwords = spmproc.spmproc.id_to_string(subword_ids).numpy().tolist()  # this contains bytes, not strings
        subwords = [s.decode() for s in subwords]
        ret = tf.argmax(ulmfit_tagger.predict(subword_ids_tensor)[0], axis=1).numpy().tolist()
        for subword, category in zip(subwords, ret):
            print("{:<15s}{:>4s}".format(subword, label_map[category]))


def train_step(*, model, hub_object, loss_fn, optimizer, awd_off=None, x, y, step_info):
    if awd_off is not True:
        if hub_object is not None: hub_object.apply_awd(0.5)
        else: apply_awd_eagerly(model, 0.5)
    with tf.GradientTape() as tape:
        y_preds = model(x, training=True)
        loss_value = loss_fn(y_true=y, y_pred=y_preds)
        print(f"Step {step_info[0]}/{step_info[1]} | batch loss before applying gradients: {loss_value}")

    grads = tape.gradient(loss_value, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))


def main(args):
    check_unbounded_training(args.get('fixed_seq_len'), args.get('max_seq_len'))
    train_jsonl = r_jsonl(args['train_jsonl'])
    label_map, rev_label_map = read_labels(args['label_map'])
    spm_args = {'spm_model_file': args['spm_model_file'],
                'add_bos': False,
                'add_eos': False,
                'lumped_sents_separator': '[SEP]',
                'fixed_seq_len': args.get('fixed_seq_len')}
    layer_config = {'qrnn': args.get('qrnn'),
                    'num_recurrent_layers': args.get('num_recurrent_layers'),
                    'qrn_zoneout': args.get('qrnn_zoneout') or 0.0}
    spmproc = LMTokenizerFactory.get_tokenizer(tokenizer_type='spm',
                                               tokenizer_file=args['spm_model_file'],
                                               fixed_seq_len=args.get('fixed_seq_len'),
                                               add_bos=False, add_eos=False)  # bos / eos will need to be added manually
    tokenized, numericalized, labels, encoded_labels = tokenize_and_align_labels(spmproc=spmproc,
                                                                                 sents=train_jsonl,
                                                                                 max_seq_len=args.get('fixed_seq_len'),
                                                                                 do_padding=True if args.get('fixed_seq_len') is not None else False,
                                                                                 rev_label_map=rev_label_map,
                                                                                 is_multilabel=args.get('multilabel'),
                                                                                 onehot_encode=True if args.get('multilabel') else False,
                                                                                 add_bos=True,
                                                                                 add_eos=True)
    r = random.randint(0, len(tokenized)-1)
    print(f"Sentence {r} and its tags:")
    for t, n, l, el in zip(tokenized[r], numericalized[r], labels[r], encoded_labels[r]):
        print(f"{t}\t{n}\t{l}\t{el}")

    print(f"Generating {'ragged' if args.get('fixed_seq_len') is None else 'dense'} tensor inputs...")
    if args.get('fixed_seq_len') is not None:
        sequence_inputs = tf.constant(numericalized, dtype=tf.uint32)
        subword_labels = tf.constant(encoded_labels, dtype=tf.uint8) # 256 labels ought to be enough for everybody
    else:
        sequence_inputs = tf.ragged.constant(numericalized, dtype=tf.uint32)
        subword_labels = tf.ragged.constant(encoded_labels, dtype=tf.uint8) # 256 labels ought to be enough for everybody
    if args.get('multilabel'):
        subword_labels = tf.cast(subword_labels, dtype=tf.bool) # no idea why tf.constant refuses to process one-hot arrays into bools directly...

    # activation and loss functions:
    if args.get('multilabel'):
        activation = 'sigmoid'
        loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=False)
        # loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=False)
    else:
        activation = 'softmax'
        loss_fn = RaggedSparseCategoricalCrossEntropy() \
                  if args.get('fixed_seq_len') is None \
                  else tf.keras.losses.SparseCategoricalCrossentropy()
    ulmfit_tagger, hub_object = ulmfit_sequence_tagger(model_type=args['model_type'],
                                                       pretrained_encoder_weights=args['model_weights_cp'],
                                                       spm_model_args=spm_args,
                                                       fixed_seq_len=args.get('fixed_seq_len'),
                                                       num_classes=len(label_map),
                                                       activation=activation,
                                                       layer_config=layer_config)

    num_steps = (sequence_inputs.shape[0] // args['batch_size']) * args['num_epochs']
    print_training_info(args=args, x_train=sequence_inputs, y_train=subword_labels)
    if args.get('lr_scheduler') == 'stlr':
        scheduler = STLRSchedule(args['lr'], num_steps)
    else:
        scheduler = args['lr']
    optimizer_fn = tf.keras.optimizers.Adam(learning_rate=scheduler)
    main_metric = 'accuracy' if args.get('multilabel') else 'sparse_categorical_accuracy'
    callbacks = prepare_keras_callbacks(args=args, model=ulmfit_tagger, hub_object=hub_object,
                                        monitor_metric=main_metric)
    if args.get('lr_scheduler') == '1cycle':
        print("Fitting with one-cycle")
        callbacks.append(OneCycleScheduler(steps=num_steps, lr_max=args['lr']))
    print(f"Shapes - sequence inputs: {sequence_inputs.shape}, labels: {subword_labels.shape}")


    save_best_path = os.path.join(args['out_path'], 'tagger_best')
    save_last_path = os.path.join(args['out_path'], 'tagger_last')
    os.makedirs(save_best_path, exist_ok=True)
    os.makedirs(save_last_path, exist_ok=True)

    ulmfit_tagger.summary()
    ulmfit_tagger.compile(optimizer=optimizer_fn, loss=loss_fn, metrics=[main_metric])

    # ##### This works only with fixed-length sequences:
    ulmfit_tagger.fit(sequence_inputs, subword_labels, epochs=args['num_epochs'], batch_size=args['batch_size'],
                      callbacks=callbacks)
    ulmfit_tagger.save_weights(save_last_path)

    ##### For RaggedTensors and variable-length sequences we have to use the GradientTape ########x

    # batch_size = args['batch_size']
    # steps_per_epoch = sequence_inputs.shape[0] // batch_size
    # total_steps = 0
    # for epoch in range(args['num_epochs']):
    #     for step in range(steps_per_epoch - 1):
    #         if total_steps % args['save_every'] == 0:
    #             print("Saving weights...")
    #             ulmfit_tagger.save_weights(save_path)
    #         train_step(model=ulmfit_tagger,
    #                    hub_object=hub_object,
    #                    loss_fn=loss_fn, optimizer=optimizer_fn,
    #                    x=sequence_inputs[(step*batch_size):(step+1)*batch_size],
    #                    y=subword_labels[(step*batch_size):(step+1)*batch_size],
    #                    step_info=(step, steps_per_epoch))
    #         total_steps += 1-


if __name__ == "__main__":
    argz = argparse.ArgumentParser()
    argz.add_argument("--train-jsonl", required=False, help="Whitespace-pretokenized and annotated input file")
    argz.add_argument("--label-map", required=True, help="Path to a JSON file containing labels.")
    argz.add_argument("--multilabel", action='store_true', help="Whether tokens can be tagged by more than one entity "
                                                                "(sigmoid, not softmax)")
    argz.add_argument("--model-weights-cp", required=True, help="Training: path to *weights* (checkpoint) of "
                                                                "the generic model (not the SavedModel/HDF5 blob!)."
                                                                "Evaluation/Interactive: path to *weights* produced "
                                                                "by this script during training")
    argz.add_argument("--qrnn", action='store_true', help="Set this if the pretrained weights contain a QRNN-based encoder, " \
                                                          "otherwise it's an ULMFiT-based model.")
    argz.add_argument("--qrnn-zoneout", type=float, help="Optional zoneout for the QRNN model.")
    argz.add_argument("--num-recurrent-layers", type=int, help="Number of recurrent layers in the encoder.")
    argz.add_argument("--model-type", choices=['from_cp', 'from_hub'], default='from_cp',
                      help="Model type: from_cp = from checkpoint, from_hub = from TensorFlow hub")
    argz.add_argument('--spm-model-file', required=True, help="Path to SentencePiece model file")
    argz.add_argument('--awd-off', required=False, action='store_true', help="Switch off AWD in the training loop.")
    argz.add_argument('--fixed-seq-len', required=False, type=int, help="Fixed sequence length. If unset, the training "
                                                                        "script will use ragged tensors. Otherwise, it "
                                                                        "will use padding.")
    argz.add_argument('--max-seq-len', required=False, type=int, help="Maximum sequence length")
    argz.add_argument("--batch-size", default=32, type=int, help="Batch size")
    argz.add_argument("--num-epochs", default=1, type=int, help="Number of epochs")
    argz.add_argument("--lr", default=0.001, type=float, help="Learning rate")
    argz.add_argument("--lr-scheduler", choices=['stlr', '1cycle', 'constant'], default='constant', help="Learning rate"
                      "scheduler (slanted triangular, one-cycle or constant LR)")
    argz.add_argument("--interactive", action='store_true', help="Run the script in interactive mode")
    argz.add_argument("--save-every", type=int, help="How often to save a checkpoint (number of steps)")
    argz.add_argument("--out-path", default="ulmfit_tagger", help="Training: Checkpoint name to save every N steps")
    argz = vars(argz.parse_args())
    if all([argz.get('max_seq_len') and argz.get('fixed_seq_len')]):
        print("You can use either `max_seq_len` with RaggedTensors to restrict the maximum sequence length, or"
              "`fixed_seq_len` with dense tensors to set a fixed sequence length with automatic padding, not both.")
        exit(1)
    if argz.get('train_jsonl') is None and argz.get('interactive') is None:
        print("Please provide either a data file for training / evaluation or run the script with --interactive switch")
        exit(0)
    if argz.get('interactive') is True:
        interactive_demo(argz)
    else:
        main(argz)
