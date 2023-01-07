import tensorflow as tf
import tensorflow_hub as hub

from .encoders import tf2_recurrent_encoder, HubRaggedWrapper, TiedDense, RaggedConcatPooler, ConcatPooler


def build_rnn_encoder_native(*, pretrained_weights=None, fixed_seq_len=None, spm_model_args,
                             also_return_spm_encoder=False, return_lm_head=False, layer_config=None):
    """ Build a MultiFiT or ULMFiT encoder using Keras layers and Python code """
    print("Building model from Python code (not tf.saved_model)...")
    lm_num, enc_num, _, spm_encoder_model = tf2_recurrent_encoder(fixed_seq_len=fixed_seq_len, spm_args=spm_model_args,
                                                                  flatten_ragged_outputs=False, layer_config=layer_config)
    if pretrained_weights is not None:
        print("Restoring weights from file....")
        lm_num.load_weights(pretrained_weights)
    else:
        print("!!! THE MODEL WEIGHTS ARE UNINITIALIZED !!! Make sure to restore them from file.")
    ret_layer = lm_num if return_lm_head is True else enc_num
    if also_return_spm_encoder is True:
        return ret_layer, spm_encoder_model
    else:
        return ret_layer


def restore_rnn_encoder_from_savedmodel(*, pretrained_weights=None, fixed_seq_len=None, spm_model_args=None,
                                        also_return_spm_encoder=False, return_lm_head=False):
    """ Restores a MultiFiT or ULMFiT encoder from a serialized SavedModel  """
    if also_return_spm_encoder:
        print(f"Info: The SPM layer is baked into the SavedModel. It will not be returned separately.")
    if spm_model_args is not None:
        print(f"Info: When restoring from a SavedModel, `spm_model_args` has no effect`")
    restored_hub = hub.load(pretrained_weights)

    if fixed_seq_len is None:
        il = tf.keras.layers.Input(shape=(None,), ragged=True, name="numericalized_input", dtype=tf.int32)
        kl_restored = HubRaggedWrapper(hub_layer=hub.KerasLayer(restored_hub.signatures['numericalized_encoder'], trainable=True),
                                       name="hub_ulmfit_encoder_ragged")
        kl_tensor = kl_restored(il)
        if return_lm_head:
            if not hasattr(restored_hub, 'lm_head_biases'):
                raise ValueError("This SavedModel was serialized without the LM head biases. Please export "
                                 "from FastAI again.")
            # rt = tf.RaggedTensor.from_row_splits(kl_restored[0], kl_restored[1])
            reference_layer = getattr(restored_hub.encoder_num, 'layer_with_weights-0')
            lm_head_ragged = tf.keras.layers.TimeDistributed(TiedDense(reference_layer, 'softmax'))
            lm_head_ragged.set_weights([restored_hub.lm_head_biases.value()]) # untested
            lm_head_ragged = lm_head_ragged(kl_tensor)
            ret_tensor = lm_head_ragged
            # kl = tf.keras.models.Model(inputs=il, outputs=lm_head_ragged)
            # kl.layers[-1].set_weights()
        else:
            ret_tensor = kl_tensor
    else:
        il = tf.keras.layers.Input(shape=(fixed_seq_len,), name="numericalized_input", dtype=tf.int32)
        kl_tensor = hub.KerasLayer(restored_hub.signatures['numericalized_encoder'], trainable=True, name="rnn_encoder")(il)['output']
        if return_lm_head:
            if not hasattr(restored_hub, 'lm_head_biases'):
                raise ValueError("This SavedModel was serialized without the LM head biases. Please export "
                                 "from FastAI again.")
            reference_layer = getattr(restored_hub.encoder_num, 'layer_with_weights-0')
            lm_head = tf.keras.layers.TimeDistributed(TiedDense(reference_layer, 'softmax'))
            lm_head.set_weights([restored_hub.lm_head_biases.value()]) # untested
            lm_head = lm_head(kl_tensor)
            ret_tensor = lm_head
        else:
            ret_tensor = kl_tensor
    model = tf.keras.models.Model(inputs=il, outputs=ret_tensor)
    return model, restored_hub


def build_sequence_tagger(*, model_type, pretrained_encoder_weights, spm_model_args=None, fixed_seq_len=None, num_classes,
                          activation, layer_config=None):
    """ Build a sequence tagger """
    if model_type == 'from_cp':
        ulmfit_rnn_encoder = build_rnn_encoder_native(pretrained_weights=pretrained_encoder_weights,
                                                      spm_model_args=spm_model_args,
                                                      fixed_seq_len=fixed_seq_len,
                                                      also_return_spm_encoder=False,
                                                      layer_config=layer_config)
        hub_object = None
    elif model_type == 'from_hub':
        raise NotImplementedError("Building a sequence tagger from a pretrained SavedModel isn't implemented yet")
    else:
        raise ValueError(f"Unknown model type {model_type}")
    tagger_head = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(num_classes, activation=activation))(ulmfit_rnn_encoder.output)
    tagger_model = tf.keras.models.Model(inputs=ulmfit_rnn_encoder.inputs, outputs=tagger_head)
    return tagger_model, hub_object


def build_document_classifier(*, model_type, pretrained_encoder_weights, num_classes,
                              spm_model_args=None, fixed_seq_len=None, use_bias=False,
                              with_batch_normalization=False, activation='softmax',
                              layer_config=None):
    """
    Document classification head as per the ULMFiT paper:
       - AvgPool + MaxPool + Last hidden state
       - BatchNorm
       - 2 FC layers
    """
    if model_type == 'from_cp':
        ulmfit_rnn_encoder = build_rnn_encoder_native(pretrained_weights=pretrained_encoder_weights,
                                                      spm_model_args=spm_model_args,
                                                      fixed_seq_len=fixed_seq_len,
                                                      also_return_spm_encoder=False,
                                                      layer_config=layer_config)
        hub_object=None

    elif model_type == 'from_hub':
        ulmfit_rnn_encoder, hub_object = restore_rnn_encoder_from_savedmodel(pretrained_weights=pretrained_encoder_weights,
                                                                             spm_model_args=None,
                                                                             fixed_seq_len=fixed_seq_len,
                                                                             also_return_spm_encoder=False)
    else:
        raise ValueError(f"Unknown model type {model_type}")

    if fixed_seq_len is None:
        rpooler = RaggedConcatPooler(name="RaggedConcatPooler")(ulmfit_rnn_encoder.output)
    else:
        rpooler = ConcatPooler(name="ConcatPooler")(ulmfit_rnn_encoder.output)

    drop_pooler = tf.keras.layers.Dropout(0.2)(rpooler)
    if with_batch_normalization is True:
        bnorm_pooler = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1, scale=False, center=False)(drop_pooler)
        bnorm_drop = tf.keras.layers.Dropout(0.1)(bnorm_pooler)
        fc1 = tf.keras.layers.Dense(50, activation='linear', use_bias=use_bias)(bnorm_drop)
        relu1 = tf.keras.layers.ReLU()(fc1)
        bnorm1 = tf.keras.layers.BatchNormalization(epsilon=1e-05, momentum=0.1, scale=False, center=False)(relu1)
        drop2 = tf.keras.layers.Dropout(0.1)(bnorm1)
        fc_final = tf.keras.layers.Dense(num_classes, use_bias=use_bias, activation='softmax')(drop2)
    else:
        fc1 = tf.keras.layers.Dense(50, activation='relu', use_bias=use_bias)(drop_pooler)
        drop2 = tf.keras.layers.Dropout(0.1)(fc1)
        fc_final = tf.keras.layers.Dense(num_classes, activation=activation, use_bias=use_bias)(drop2)

    document_classifier_model = tf.keras.models.Model(inputs=ulmfit_rnn_encoder.inputs, outputs=fc_final)
    return document_classifier_model, hub_object


def build_regressor(*, model_type, pretrained_encoder_weights, spm_model_args=None, fixed_seq_len=None,
                    with_batch_normalization=False, layer_config=None):
    """
    Regression head which outputs a single numerical value. The architecture is similar to
    the document classification head and differs only in the last layer (a single neuron
    with no activation instead of softmax).
    """
    regressor, hub_object = build_document_classifier(model_type=model_type,
                                                      pretrained_encoder_weights=pretrained_encoder_weights,
                                                      num_classes=1,
                                                      spm_model_args=spm_model_args,
                                                      fixed_seq_len=fixed_seq_len,
                                                      use_bias=True,
                                                      with_batch_normalization=with_batch_normalization,
                                                      activation='linear',
                                                      layer_config=layer_config)
    return regressor, hub_object
