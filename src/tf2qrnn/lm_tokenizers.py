import sentencepiece as spm

from ulmfit_tf2 import SPMNumericalizer


class LMTokenizerFactory:
    """ Tokenizer factory

        Creates sentencepiece tokenizer objects from pretrained models.
        The `tokenizer_type` parameter controls the returned object's type:
           - 'spm'         - returns a SentencePieceProcessor object
           - 'spm_tf_text' - returns a Keras layer that wraps the tensorflow_text.SentencepieceTokenizer object
    """
    def __init__(self):
        pass

    @staticmethod
    def get_tokenizer(*, tokenizer_type, tokenizer_file, add_bos=False, add_eos=False, fixed_seq_len=None):
        if tokenizer_type == 'spm':
            tok_obj = spm.SentencePieceProcessor(tokenizer_file)
            extra_opts = []
            # Setting the ._add_bos or ._add_eos fields has no effect - the only way to modify sentencepiece object's
            # behavior is to pass these settings as extra options. However, we do set these fields anyway
            # for sanity checks in the sequence tagger.
            if add_bos: extra_opts.append("bos"); tok_obj._add_bos = True
            if add_eos: extra_opts.append("eos"); tok_obj._add_eos = True
            tok_obj.set_encode_extra_options(":".join(extra_opts))
        elif tokenizer_type == 'spm_tf_text':
            tok_obj = SPMNumericalizer(spm_path=tokenizer_file, add_bos=add_bos, add_eos=add_eos,
                                       fixed_seq_len=fixed_seq_len)
        elif tokenizer_type == 'none':
            tok_obj = None
        else:
            raise ValueError(f"Unknown tokenizer type {tokenizer_type}")
        return tok_obj
