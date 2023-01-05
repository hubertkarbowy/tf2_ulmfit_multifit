"""
Various ULMFit / FastAI related utils
"""
import os

from fastai.text.data import TensorText
from fastcore.foundation import L

from ulmfit_commons import file_len
from ulmfit_tf2 import tf2_ulmfit_encoder


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
            num_sents = file_len(datasource_path)
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

def save_as_keras(*, state_dict, exp_name, save_path, spm_model_file, fixed_seq_len, layer_config):
    """
    Creates an ULMFit inference model using Keras layers and copies weights from FastAI's learner.model.state_dict() there.

    There are many explicit constants in this function, which is intentional. The numbers 400, 1152 and 3 layers refer
    to the paper's implementation of ULMFit in FastAI.

    """
    spm_args = {
        'spm_model_file': spm_model_file,
        'add_bos': True,
        'add_eos': True,
        'lumped_sents_separator': '[SEP]'
    }
    lm_num, encoder_num, outmask_num, spm_encoder_model = tf2_ulmfit_encoder(fixed_seq_len=fixed_seq_len, spm_args=spm_args,
                                                                             layer_config=layer_config)
    lm_num.summary()
    lm_num.get_layer('ulmfit_embeds').set_weights([state_dict['0.encoder.weight'].cpu().numpy()])

    for i_rnn in range(layer_config['num_recurrent_layers']):
        if layer_config['qrnn']:
            rnn_weights = [state_dict[f'0.rnns.{i_rnn}.layers.0.linear.weight_raw'].cpu().numpy().T,
                           state_dict[f'0.rnns.{i_rnn}.layers.0.linear.module.bias'].cpu().numpy()]
            keras_layer = lm_num.get_layer(f'QRNN{i_rnn+1}')
            print(f"FastAI shapes: {rnn_weights[0].shape} and {rnn_weights[1].shape}")
            print(f"Keras shapes: {keras_layer.variables[0].shape} and {keras_layer.variables[1].shape}")
            keras_layer.set_weights(rnn_weights)

        else:
            rnn_weights = [state_dict[f'0.rnns.{i_rnn}.module.weight_ih_l0'].cpu().numpy().T,
                           state_dict[f'0.rnns.{i_rnn}.weight_hh_l0_raw'].cpu().numpy().T,
                           state_dict[f'0.rnns.{i_rnn}.module.bias_ih_l0'].cpu().numpy()*2]
            lm_num.get_layer(f'AWD_RNN{i_rnn+1}').set_weights(rnn_weights)

    # rnn_weights1 = [state_dict['0.rnns.0.module.weight_ih_l0'].cpu().numpy().T,
    #                 state_dict['0.rnns.0.weight_hh_l0_raw'].cpu().numpy().T,
    #                 state_dict['0.rnns.0.module.bias_ih_l0'].cpu().numpy()*2]
    # rnn_weights2 = [state_dict['0.rnns.1.module.weight_ih_l0'].cpu().numpy().T,
    #                 state_dict['0.rnns.1.weight_hh_l0_raw'].cpu().numpy().T,
    #                 state_dict['0.rnns.1.module.bias_ih_l0'].cpu().numpy()*2]
    # rnn_weights3 = [state_dict['0.rnns.2.module.weight_ih_l0'].cpu().numpy().T,
    #                 state_dict['0.rnns.2.weight_hh_l0_raw'].cpu().numpy().T,
    #                 state_dict['0.rnns.2.module.bias_ih_l0'].cpu().numpy()*2]

    # lm_num.get_layer('AWD_RNN1').set_weights(rnn_weights1)
    # lm_num.get_layer('AWD_RNN2').set_weights(rnn_weights2)
    # lm_num.get_layer('AWD_RNN3').set_weights(rnn_weights3)
    keras_lm_head_name = 'lm_head_tied' if layer_config['num_hidden_units'][-1] == layer_config['emb_dim'] \
                         else 'lm_head_untied'

    lm_num.get_layer(keras_lm_head_name).set_weights([state_dict['1.decoder.bias'].cpu().numpy(),
                                                      state_dict['1.decoder.weight'].cpu().numpy()])
    lm_num.save_weights(os.path.join(save_path, exp_name))
    return lm_num, encoder_num, outmask_num, spm_encoder_model
