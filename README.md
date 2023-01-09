# QRNN (Quasi-Recurrent Neural Network) and ULMFiT for Tensorflow 2.x

Since the arrival of BERT, recurrent neural networks have fallen out of fashion in Natural Language Processing. In practical applications, however, there are still many use cases where a much lighter recurrent architecture is preferrable over the Transformer due to e.g. faster inference times or constraints on the computing power. This repo contains the implementation of two such recurrent architectures in Tensorflow:

* **[ULMFiT](https://aclanthology.org/P18-1031/)** - which is basically LSTM with improvements to the regularization techniques and a triangular learning rate scheduler,
* [**MultiFIT**](https://aclanthology.org/D19-1572) - the evolution of the ULMFiT model in which LSTM cells were replaced by Quasi-Recurrent cells and a One-cycle learning rate scheduler was used.

Both architectures were originally implemented in the [FastAI](https://www.fast.ai/) framework which itself is build over [PyTorch](https://pytorch.org/). However, the implementations were poorly documented, examples were scarce and adaptation to custom data was challenging. FastAI's text processing pipeline is also rather inflexible. Subjectively, the two biggest issues are the difficulty in using subword tokenization and the baking in as defaults of some rather arcane transformations (for example, capital letters are downcased, but a `\\xxmaj` symbol is inserted before to denote capitalization). These issues can for sure be overcome by reverse-engineering FastAI code where the documentation remains silent. 

Instead, a cleaner implementation of both architectures, less tightly coupled with the data preprocessing techniques is provided here for Tensorflow, which is also more familiar to the Machine Learning community and has a bigger user base than FastAI. 

Table of contents

[TOC]



## 1. Introduction

### 1.1. Installation and quick start

No installation is required if you are planning to use only the [pretrained encoders](#2.-pretrained-models-and-corpora) from SavedModel files. Download one of them and extract it to a directory:

```
$ mkdir sample_model
$ wget https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EWtQ0LyOzfBNoyprcId2D6kBpD2zkjK9buTpjzOztSBIcw?download=1 -O sample_model/m.tgz
$ tar xvfz ./sample_model/m.tgz -C ./sample_model/
```

Then, in Python run:

```
import tensorflow as tf
import tensorflow_text # needed to register sentencepiece graph operations

model = tf.saved_model.load('./sample_model/')
sents = tf.constant(['the limits of my language mean the limits of my world'])
model.signatures['string_encoder'](sents)

{'output': <tf.Tensor: shape=(1, 70, 400), dtype=float32, numpy=
 array([[[-0.06846653,  0.03071258,  0.04870525, ...,  0.04809854,
           0.159233  , -0.02486314],
         [-0.09425692,  0.11809141,  0.00196292, ...,  0.06615587,
           0.12720759,  0.14610538],
         [-0.06734478,  0.02118873,  0.08690275, ..., -0.26142147,
          -0.10768991,  0.14759885],
         ...,
         [ 0.02605111, -0.01863609,  0.04656828, ..., -0.0163744 ,
          -0.1731047 , -0.1022887 ],
         [ 0.02605111, -0.01863609,  0.04656828, ..., -0.0163744 ,
          -0.1731047 , -0.1022887 ],
         [ 0.02605111, -0.01863609,  0.04656828, ..., -0.0163744 ,
          -0.1731047 , -0.1022887 ]]], dtype=float32)>,
 'numericalized': <tf.Tensor: shape=(1, 70), dtype=int32, numpy=
 array([[   2,   10, 6549,   28, 1575, 1240, 1573,   10, 6549,   28, 1575,
          778,    3,    1,    1,    1,    1,    1,    1,    1,    1,    1,
            1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
            1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
            1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
            1,    1,    1,    1,    1,    1,    1,    1,    1,    1,    1,
            1,    1,    1,    1]], dtype=int32)>,
 'mask': <tf.Tensor: shape=(1, 70), dtype=bool, numpy=
 array([[ True,  True,  True,  True,  True,  True,  True,  True,  True,
          True,  True,  True,  True, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False, False, False,
         False, False, False, False, False, False, False]])>}
```

The token vectors are under the `output` key. As you can see, the input sentences are automatically tokenized into wordpieces, converted to IDs ("numericalized") and padded to a sequence length of 70. There is quite a lot going on here and we'll explain the details later in this guide.

To run the examples for document classification, sequence tagging etc. from Section 4 and 5 you need to install this repo via pip:

```
$ pip install tf2qrnn@git+https://github.com/hubertkarbowy/tf2qrnn.git
```

Note, however, that the requirements **DO NOT INCLUDE** the dependencies needed for FastAI-related scripts - these are so outdated that including them would prevent any use of this package in a modern project. If needed (see next section), you will need to install them manually probably in a separate environment.



### 1.2. But do we still need FastAI? And why such an antique version?

TLDR: Yes, but only if you want to pretrain a Language Model from scratch.

FastAI is still needed for pretraining a language model from scratch or fine-tuning it to a specific domain. It is, however, not needed if you are happy with the models pretrained on the general languages corpora listed below and your only concern is the target task such as document classification or sequence tagging - in this case Tensorflow alone will suffice.

Why is that? Essentially because ULMFiT and MultiFiT architectures were developed by FastAI authors and associates and the training optimizations used are not easily portable to another machine learning framework. In particular, FastAI was the only framework that not only natively supported QRNNs, but also had custom CUDA code that provided acceleration on a GPU. Sadly, in FastAI v. 2.4 "*QRNN module [was] removed, due to incompatibility with PyTorch 1.9, and lack of utilization of QRNN in the deep learning community.*" (from their release notes). For quasi-recurrent networks, therefore, we are stuck with those old libraries and not much can be done about it.

The classes in this repo are built on top of [Keras RNN API](https://www.tensorflow.org/guide/keras/rnn). This allows AWD LSTM / QRNN cells to be used as a drop-in replacement for LSTM or GRU cells available in Keras. However, unlike the latter two, there is no CUDA-optimized version for a QRNN - this would require writing a new kernel and making it work with TF, which is far from trivial. That is quite a drawback if one was to train a language model from scratch - the code in old FastAI version optimized for that task runs by several orders of magnitude faster than generic CUDA kernels used by Tensorflow. For language modelling the data runs into gigabytes, so the acceleration is a must.

However, when using transfer learning, models for downstream tasks such as sentiment analysis or named entity recognition can be trained to acceptable performance on much smaller volumes of data. Therefore, we use FastAI to pretrain only the language model (the encoder), then we export its weights and read them into and identically structured Keras equivalent. From this point, we can run the far less computation-intensive downstream task with only generic CUDA optimizations provided by Tensorflow and still finish the training in reasonable time.

The script used to pretrain ULMFiT and MultiFiT language models in FastAI is called [convert_fastai2keras.py](src/tf2qrnn/convert_fastai2keras.py) - its usage and conversion to Keras is discussed below.



### 1.3. Scope of reimplementation

We were successful in porting the following regularization and training techniques used in ULMFiT and MultiFiT to TF2:

* encoder dropout
* input dropout
* RNN dropout
* weight dropout (AWD, must be called manually or via a KerasCallback) for LSTM and zoneout for QRNN
* slanted triangular learning rates (available as a subclass of  `tf.keras.optimizers.schedules.LearningRateSchedule`) and one-cycle policy together with the learning rate finder ([implementations by Andrich van Wyk](https://www.kaggle.com/avanwyk/tf2-super-convergence-with-the-1cycle-policy)).

The following techniques were not ported:

* gradual unfreezing - but you can very easily control this yourself by setting the `trainable` attribute on successive Keras layers either manually or by writing a simple callback.
* some mysterious undocumented callbacks in FastAI such as `rnn_cbs(alpha=2, beta=1)`
* batch normalization



## 2. Pretrained models and corpora

Several pretrained models for English and Polish are made available in the following formats:

* **TF 2.0 SavedModel** - these are self-contained, standalone binaries that contain serialized assets, graphs, functions and weights for the three components needed to obtain token vectors produced by the language model:

  * Sentencepiece tokenizer and subword vocabulary models
  * Numericalization, i.e. the transformation of word pieces into index IDs
  * The encoder itself

  This is probably the easiest format to use since you don't need any code from this repo. However, it comes with certain limitations, the most significant of which is that the sequence length is fixed at 70 tokens and that the `<s>` and `</s>` symbols are added automatically for each textual input.

* **Keras weights** - these are checkpoint files that can be restored via `model.load_weights(...)` on a Keras object instantiated using code from this repo. This can be handy if you need to tweak some parameters that were fixed by the paper's authors, for example add or remove some dropout layers.

* **FastAI .pth** **state_dict** - the file containing the encoder trained using FastAI as explained above. Both the SavedModel and Keras weights formats were generated from this file using the `convert_fastai2keras.py` script.

Here is the complete list of available pretrained models:

<u>**English (trained on the Wikitext corpus)**</u>

All models were trained on a corpus similar to [Wikitext-103](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) with a subword dictionary of 35,000 tokens. A fresh Wikipedia dump was prepared using the [wikiextractor](https://github.com/attardi/wikiextractor) package, then the output was sentence-tokenized and underwent some further minor preprocessing using the `01_cleanup.py` script. If needed, the corpus files (split into training, validation and test sets) can be downloaded from these links:

* [uncased](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EbVZzpeXKO1BqQjBtUfxXTgBee7L6746EazGph4k2-cSHQ?download=1)
* [cased](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EShjqqOiiihMgo33mL5iuO8Bw9Dfq_OG-laVV4di86fcyA?download=1)

| Model                         | TF 2.0 SavedModel                                            | Keras weights                                                | FastAI v2.2.7 model                                          | Sentencepiece files                                          |
| ----------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **ulmfit-en-sp35k-cased**     | [.tar.gz archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EWtQ0LyOzfBNoyprcId2D6kBpD2zkjK9buTpjzOztSBIcw?download=1) | [.tar.gz archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/ERWfQcH7Sj1IqViKl2-ygRsBxryv7FfJ-rVcpPWkjaOt6g?download=1) | [.pth file](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EdZPByYtSp5Jkfa1MLRKjPcBJiQ3XSe94BBc56I8Lcmn-g?download=1) | [model](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EZSXfz0XKf5Jlc9nqj2kt50BZiLlFLfR4cwZh4euDEfMCw?download=1), [vocab](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EUxMYcwQ4gpPjR32V1MR9BUBdGYgHRKS7w7daM0ebkwc7A?download=1) |
| **ulmfit-en-sp35k-uncased**   | [.tar.gz archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EZtBWV10etNKkqx3YXGDdqMBiIlAGNEKNuxCrSPZsqoKeA?download=1) | [.tar.gz archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EaTWKKlapRBOg88am0Z0j6sBleNntd_ETAEo_E8l18E8hA?download=1) | [.pth file](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/Ef30f7aZTGVOoqja4BIchDUBz60fLZj8Gpw-ttS9rZVZjg?download=1) | [model](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EYqvtu342mlNgCoO6O9_S1ABfN-dv6uQNI8aO5zsj0MT9g?download=1), [vocab](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EZeSclPihexKjXo4dMiLxTkBFBQPCQF6DsAmI-GYB0aT1A?download=1) |
| **multifit-en-sp35k-cased**   | [.tar.gz archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EY9Q05bd695LkfpVUrnGfXcB1yMiltINkwuRM_8FKSk2mQ?download=1) | [.tar.gz archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/ESn5s-KMmAxOnQl8TJRwKG8BAPa1zd40oi5e5Vne4lbRHA?download=1) | [.pth file](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EaIpprBIfypKtosmuQbFU0UBTXN5okDRc096WHsy6gHFIw?download=1) | [model](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EVdQVGOSgV5MujoBWVb_4EoB1CgIbFsoxvnKz299dF8jAg?download=1), [vocab](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EbAvZwe9tOlCsiwA0cOsTJQBa6_Y-1UkXawWTe2iozk1Jg?download=1) |
| **multifit-en-sp35k-uncased** | [.tar.gz archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EQABorkyinVBi3OAJHtQjGMBD4mWPSd4-tN8z7iHUvm0hQ?download=1) | [.tar.gz archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EToP5dl5HpdLudZbleCM4A0BxJ8nFLMAPtCRm204Bx8tnQ?download=1) | [.pth file](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EYYr_Q3r-kJNhmb-wbRzIxIBpFz2edCSucXjXUIpKqW_JA?download=1) | [model](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EX8noqB8teZAm5JGmYViAq4BR_zjAyJ6PjUOtWqnD5g2dw?download=1), [vocab](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/ERAQq9qQu5RKvk-uf37mVDcBQMcUSFB2m83z-fUZWXvP8A?download=1) |



**<u>Polish (trained on the Wikitext corpus):</u>**

These models are available with subword dictionary sizes of 35,000 and 50,000 tokens (both cased). The corpus links are:

* [cased](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EWyV047JBwBCvUDjKVCwNKEB2yeX5RRn2t2Qq6w4XBoS_w?download=1)
* [uncased](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EQoS9MIBU8NEk2yzya_kT8wB4Ei7wN5WZ_zxw-khEHkUZA?download=1) (no models provided)

| Model                       | TF 2.0 SavedModel                                            | Keras weights                                                | FastAI v2.2.7 model                                          | Sentencepiece files                                          |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **ulmfit-pl-sp35k-cased**   | [.tar.gz archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EVREU5kXGVNIqr6v3T5jL0UBsT5ujoKcOiW2dsZFUUnUNQ?download=1) | [.tar.gz archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EcYa-n6JFdxBnnRVSpcKvO4B6KsJrK50ET_HvJI8mRXfiA?download=1) | [.pth file](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/ESHg6JCkiHJCqeTOP6lY7OABae72sXp5p53XU1dlMHCqKw?download=1) | [model](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/ET57COd7jkhBiKC_Nmc4zLMB_v7pB4PdaHTRRyNda3btJQ?download=1), [vocab](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/Efq36t20Yx5Bj1vr-puCmZgB33KIoNKKz7JphkS5J4rbXA?download=1) |
| **ulmfit-pl-sp50k-cased**   | [.tar.gz archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EfEX_YUu1q5Cglf_vUwDBHkB-fM0N5q-pvbnfqIVxwvSzQ?download=1) | [.tar.gz archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/ETrxh1xqouJPvfVlkoINmycBAeXxMK0RMfVS5940JMw_KA?download=1) | [.pth file](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/ERJpDTyFeWdFkIRDIFdMurQBnJXYmIBnCYyt_nOFoqexzQ?download=1) | [model](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EaQmC0qDqzhAkcJXlbRU8kIBWHa0NUUw_sMnv9xnOAYnQw?download=1), [vocab](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EXZwxnBhVs5BnTdrOJf1CKABrH61ElRYp-xL5R2Yq6Y2BQ?download=1) |
| **multifit-pl-sp35k-cased** | [.tar.gz archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EXC9yxntr8pJrgFbyQOTS0IBXQRvRLlGbem8FM4wSiulfw?download=1) | [.tar.gz archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/ESKYx7Th-XdMr0z3ybmJXXcB8P6ammwcDWYzyzuE4jjSCg?download=1) | [.pth file](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EVM6Q4unNhJChjA5xy_sNR8BQAm3y_1UMhyRVlhqWgKSPA?download=1) | [model](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/ESJisYJx8odMhafXckwJ51sBcRzh1xqcMt2juBu_mich9Q?download=1), [vocab](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EZ18YU53dKBFhFmFLzd1NvYBr5ZDOPPM323qYGMeKSy2ew?download=1) |
| **multifit-pl-sp50k-cased** | [.tar.gz archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/ESR2JOC2tXpJicvrgsqEIoMB2H7LC3ZKtzMorUAQVRk_8Q?download=1) | [.tar.gz archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/ERMANByB4zRAhIr0Yh1bP5AB9XV9y11ZlcOzn5G64cdexQ?download=1) | [.pth file](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EY9RsoswwzJJocMJHLiqWnMBMavZ_DNOXIGw32CrUuhA1Q?download=1) | [model](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EQtzbCmLe_BIrhaOWf5fWewBQzwE0-F_KSdafEWuDTQRKg?download=1), [vocab](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EUxnS4EFePNBrWUtCIDmdWkBofrj4bQ0JJzUGUlGFDM4Wg?download=1) |



**<u>Polish (trained on the Poleval-2018 corpus):</u>**

Same dictionary parameters as for the Wikitext models. The PolEval-2018 corpus can be downloaded from their [official website](http://2018.poleval.pl/index.php/tasks/) and is already sentence-tokenized.

| Model                       | TF 2.0 SavedModel                                            | Keras weights                                                | FastAI v2.2.7 model                                          | Sentencepiece files                                          |
| --------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| **ulmfit-pl-sp35k-cased**   | [.tar.gz.archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/ERzqLNbvjmlAo3TeK2UDvr4BTe7SIIioLSUQ30RtLlEUxA?download=1) | [.tar.gz.archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EVRtfGBRkLFAh68ECJKI88MBY0GAV3XevWzVhZIO2G0gwA?download=1) | [.pth file](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EYtWHSv3xAxItoYVRBeyxQUB6puwM5N9hgKQF7C27PiPPQ?download=1) | [model](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EQqAYSzbE15JngjXIh4G_yYBQGcXkTEVwWwKcDNUpjA3_Q?download=1), [vocab](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EQEonLZ8S2RJvyUQG_J1DKABOfEqcTqbtbdaIawV8Nj1aQ?download=1) |
| **ulmfit-pl-sp50k-cased**   | [.tar.gz.archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EYoJFKwVI9tEpdBsoTqwDxABo3MtLIOOUitzWLSJzntITw?download=1) | [.tar.gz.archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EeXs-LRWM7RFoMiSadyQTqUBX9i_D_PBv9tbWxG_ukYZgw?download=1) | [.pth file](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EciL48bD-9hJhWDX90doSZQBv064Gm5RMHkj_pilPAi70w?download=1) | [model](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EQoEb_Y5pLtPl_DkM6Ogl7cBOPYxbwZ3kSpPVpJ07iPc4Q?download=1), [vocab](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EXsHSmi7UgNArvB6h5iznjcBBKiJv4KOYSZQV2hvHGK98w?download=1) |
| **multifit-pl-sp35k-cased** | [.tar.gz.archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EYeWxutR2NNHvDW1oXS_M8IBsHfMk-iZrQLhzr9_ATkn-A?download=1) | [.tar.gz.archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EWlVBXPCOWVCjkgbuNLqOgUBJxp1-1mYq0Xkba2Sgh4oUQ?download=1) | [.pth file](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EROlxuwJAn1FpPy-XoVeDQMBL0_5Q-ukTUR_42UExq6TqA?download=1) | [model](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EUEb-y_irUVFnGnoVz7BPiwBJWfmVAhtqmeLBNoFPOu8uw?download=1), [vocab](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EXQ9Wb6tqnNAhXhxjibByYUBWbMPuk8pEDNMzVrUsxIadg?download=1) |
| **multifit-pl-sp50k-cased** | [.tar.gz.archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/ESIk07vEc-dPgYzjDqbuRPoB6OXDZSMxzOWA9HLBXU433A?download=1) | [.tar.gz.archive](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/Ea9Qh92jbttAuWemMaUcM10BF71SJZIb046nIND70srlKA?download=1) | [.pth file](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EUALi3cbCKxOkU1lqdn62QIBqHjS72SXKa7ztR89bjAAUg?download=1) | [model](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EXTmciLr8CRMmsOS-h4UXCwBlHiuREkITTiQz44Ot52XqQ?download=1), [vocab](https://wsisizwit-my.sharepoint.com/:u:/g/personal/karbowy_office_wit_edu_pl/EWR6JtxrdRZPnbmV6sT57oYBUlj8kQ91bBjJ4OmIrwpuGA?download=1) |





## 3. The encoder

The encoder transforms batches of strings into sequences of vectors representing sentences. Each token is represented as a 400-dimensional vector in the encoder's output.



### 3.1. Obtaining the RNN encoder from a SavedModel

As explained in "Installation and quick start", you don't need code from this repo to restore an encoder from a SavedModel directory. However, serialization detaches a very important piece of information from the output vectors - namely the `._keras_mask` property. This is somewhat inconvenient because we cannot use such "deficient" tensors with convenient Keras layers such as TimeDistributed or Attention that rely on this property. Fortunately, this property can be restored by wrapping the encoder into a `KerasLayer` object from the `tensorflow_hub` package and a few other operations involving the `mask` key from the `string_encoder` signature . Here is a complete example:

```
import tensorflow as tf
import tensorflow_text # must do it explicitly to register the sentencepiece op
import tensorflow_hub as hub

restored_model = tf.saved_model.load('path-to-unpacked-savedmodel-directory')

input_layer = tf.keras.layers.Input((), dtype=tf.string)
encoder_layer = hub.KerasLayer(restored_model.signatures['string_encoder'], trainable=True)
vectors, mask_from_models = encoder_layer(input_layer)['output'], encoder_layer(input_layer)['mask']
mask_reshaper = tf.keras.layers.Lambda( \
    lambda mask_tensor: tf.cast(tf.expand_dims(mask_tensor, -1), dtype=tf.float32),name="mask_reshaper"\
)
reshaped_mask_tensor = mask_reshaper(mask_from_models)
zeroed_vectors = tf.keras.layers.Multiply()([vectors, reshaped_mask_tensor])
masked_vectors = tf.keras.layers.Masking(mask_value=0.0)(zeroed_vectors)

model = tf.keras.models.Model(inputs=input_layer, outputs=masked_vectors)
```

You can now run `model(tf.constant(['Hello, world!']))` and verify that the `._keras_mask` property has been restored.



### 3.2. Initializing a new encoder from Python code and loading weights from a Keras checkpoint

The code below will instantiate a fresh Keras model for a QRNN network with four layers and weights initialized randomly. The layer hyperparameters can be adjusted by modifying the dictionary returned by `tf2qrnn.get_rnn_layers_config` (which see). You will also need one of the Sentencepiece models to set up the subword tokenizer:

```
from tf2qrnn import tf2_recurrent_encoder
from tf2qrnn.commons import get_rnn_layers_config

layer_config = get_rnn_layers_config({'qrnn': True}) 
spm_args = {'spm_model_file': 'some/path/to/enwiki100-cased-sp35k.model',
            'add_bos': True,
            'add_eos': True,
            'fixed_seq_len': 70}
_, encoder_num, _, spm_encoder_model = tf2_recurrent_encoder(spm_args=spm_args,
																		 fixed_seq_len=70,
                                                                         layer_config=layer_config)

```

This function returns two objects that are of interest to us (both of them instances of `tf.keras.Model`):

* **encoder_num** - the encoder which outputs just the vectors for the topmost recurrent layer
* **spm_encoder_model** - the subword tokenizer together with numericalization. This object also ensures that sequences are padded and that the padding mask is propagated through successive Keras layers built on top of its output.

You can view their structure just like any other Keras model by calling the `summary` method:

```
encoder_num.summary()
Model: "model_2"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
numericalized_input (InputLa [(None, 70)]              0
_________________________________________________________________
ulmfit_embeds (CustomMaskabl (None, 70, 400)           14000000
_________________________________________________________________
emb_dropout (EmbeddingDropou (None, 70, 400)           0
_________________________________________________________________
inp_dropout (SpatialDropout1 (None, 70, 400)           0
_________________________________________________________________                              
QRNN1 (RNN)                  (None, 70, 1552)          3729456                                 
_________________________________________________________________
rnn_drop1 (SpatialDropout1D) (None, 70, 1552)          0
_________________________________________________________________                              
QRNN2 (RNN)                  (None, 70, 1552)          7230768
_________________________________________________________________
rnn_drop2 (SpatialDropout1D) (None, 70, 1552)          0
_________________________________________________________________                              
QRNN3 (RNN)                  (None, 70, 1552)          7230768
_________________________________________________________________                              
rnn_drop3 (SpatialDropout1D) (None, 70, 1552)          0
_________________________________________________________________
QRNN4 (RNN)                  (None, 70, 400)           1863600
_________________________________________________________________
rnn_drop4 (SpatialDropout1D) (None, 70, 400)           0
=================================================================                              
Total params: 34,054,592
Trainable params: 34,054,592
Non-trainable params: 0
```

Let's now see how we can get the sentence representation. First, we need some texts converted to token IDs. We can use the **`spm_encoder_model`** to obtain it, then all we need to do is pass these IDs to **encoder_num**:

```
text_ids = spm_encoder_model(tf.constant(['Cat sat on a mat', 'And spat'], dtype=tf.string))
vectors = encoder_num(text_ids)
print(vectors.shape)      # (2, 70, 400)
```

There are two sentences in our batch, so the zeroeth dimension is 2. Sequences are padded to 70 tokens, hence the first dimension is 70. Finally, each output hidden state is represented by 400 floats, so the third dimension is 400 (if the encoder was instantiated with `fixed_seq_len=None`, the output would instead be a `RaggedTensor` of shape `(2, None, 400)`).

You now have a encoder model with randomly initialized weights. Sometimes this is sufficient for very simple tasks, but generally you will probably want to restore the pretrained weights. This can be done using standard Keras function `load_weights` in the same way as you do with all other Keras models, for example:

```
encoder_num.load_weights('path-to-unpacked-keras-weights-dir/qrnn-enwiki100-sp35k-uncased')
```

As the optimizer state wasn't saved with the weights, you might want to add `.expect_partial()` to the above to quench warnings about that.



### 3.3. More about subword tokenization and numericalization

Unlike FastAI, we use [Sentencepiece](https://github.com/google/sentencepiece) to tokenize the input text into subwords. To convert tokens into their IDs (numericalization) we use the pretrained vocabulary files directly with Python's `sentencepiece` module and its Tensorflow wrapper available in `tensorflow_text` as described in [this manual](https://www.tensorflow.org/tutorials/tensorflow_text/subwords_tokenizer). Our vocabularies contain the following special symbols:

* 0 - `<unk>`
* 1 - `<pad>` (please note the important difference: Keras uses `0` as the default padding index, not 1. Models in this repo are already configured to have a mask on `1` propagated throughout the layers [as explained in this article](https://www.tensorflow.org/guide/keras/masking_and_padding), but should you choose to reconfigure the layers or use custom heads over the encoder, this needs to be taken into account.
* 2 - `<s>` - beginning of sentence
* 3 - `</s>`- end of sentence

We also provide a Keras layer object called `SPMNumericalizer` which you can instantiate with a path to the `.model` file. This is convenient if you just need to process a text dataset into token IDs. 

```
import tensorflow as tf
from tf2qrnn import SPMNumericalizer

spm_processor = SPMNumericalizer(name='spm_layer',
                                 spm_path='enwiki100-cased-sp35k.model',
                                 add_bos=True,
                                 add_eos=True,
                                 fixed_seq_len=70)
print(spm_processor(tf.constant(['Hello, world'], dtype=tf.string)))
<tf.Tensor: shape=(1, 70), dtype=int32, numpy=
array([[    2, 34934,     0,  9007, 34955,   482,     3,     1,     1,
            1,     1,     1,     1,     1,     1,     1,     1,     1,
            1,     1,     1,     1,     1,     1,     1,     1,     1,
            1,     1,     1,     1,     1,     1,     1,     1,     1,
            1,     1,     1,     1,     1,     1,     1,     1,     1,
            1,     1,     1,     1,     1,     1,     1,     1,     1,
            1,     1,     1,     1,     1,     1,     1,     1,     1,
            1,     1,     1,     1,     1,     1,     1]], dtype=int32)>
```

As an aside, it is possible to omit the `fixed_seq_len` parameter, in which case the output is:

```
<tf.RaggedTensor [[2, 34934, 0, 9007, 34955, 482, 3]]>
```

The code in this repo should generally work with ragged tensors, but we have found that extra effort needs to be taken when serializing and deserializing such models. Moreover, we encountered performance issues with RaggedTensors with older TF versions. Therefore, in this guide we assume the sequences are fixed-length and padded.



## 4. How to use QRNN for Tensorflow with some typical NLP tasks

In the [examples](src/tf2qrnn/examples) directory we are providing training scripts which illustrate how the QRNN / ULMFiT encoder can be used for a variety of downstream tasks. All the scripts are executable from the command line as Python modules (for example: `python -m tf2qrnn.examples.classifer --help`). After training models on custom data, you can run these scripts with the `--interactive` switch which allows you to type text in the console and display predictions. These convenience scripts were checked to run with encoder weights loaded from a Keras checkpoint (`--model-type from_cp` parameter) - we haven't checked them with encoders restored from a SavedModel.

### 4.1. Common parameter names used in this repo

The `main` method of each example script accepts a single parameter called `args` which is basically a configuration dictionary created from arguments passed in the command line. Here is a list of the most common arguments you will encounter:

* Data:
  * `--train-tsv / -- test-tsv ` - paths to source files containing training and test/validation data. For classification/regression tasks the input format is a TSV file with a header. For sequence tagging see below.
  * `--data-column-name` - name of the column with input data
  * `--gold-column-name` - name of the column with labels
  * `--label-map` - path to a text (classifier) or json (sequence tagger) file containing labels.
* Tokenization and numericalization:
  * `--spm-model-file` - path to a Sentencepiece .model file
  * `--fixed-seq-len` - if set, input data will be truncated or padded to this number of tokens. If unset, variable-length sequences and RaggedTensors will be used
  * `--max-seq-len` - maximal number of tokens in a sequence. This should generally be set to some sensible value for your data, even if you use RaggedTensors, because one maliciously long sequence can cause an OOM error in the middle of training.
* Restoring model weights and saving the finetuned version:
  * `--model-weights-cp` - path to a local directory where the pretrained encoder weights are saved
  * `--model-type` - what to expect in the directory given in the previous parameter. If set to `from_cp`, the script will expect Keras weights (in this case, provide the checkpoint name as well). If set to `from_hub`, it will expect SavedModel files.
  * `--out-path` - where to save the model's weights after the training completes.
* Training:
  * `--num-epochs` - number of epochs
  * `--batch-size` - batch size
  * `--lr` - peak learning rate for the slanted triangular learning rate scheduler
  * `--lr-scheduler` - `stlr`for slanted triangular learning rates or `1cycle` for one-cycle policy
  * `--lr-finder [NUM_STEPS]` - will run a learning rate finder for NUM_STEPS, then display a chart showing how the LR changes wrt loss
  * `--awd-off` - disables AWD regularization
  * `--save-best` - if set, the training script will save the model with the best accuracy score on the test/validation set



### 4.2. Document classification (the ULMFiT / MultiFiT way - `classifier.py`)

This script attempts to replicate the document classifier architecture from the original ULMFiT paper. On top of the encoder there is a layer that concatenates three vectors:

* the max-pooled sentence vector
* the average-pooled sentence vector
* the encoder's last hidden state

This representation is then passed through a 50-dimensional Dense layer. The last layer has a softmax activation and many neurons as there are classes. One issue we encountered here is batch normalization, which is included in the original paper and the FastAI text classifier model, but which we were not able to use in Tensorflow. When adding BatchNorm to the model we found that we could not get it to converge on the validation set, so it is disabled in our scripts. If you nevertheless wish to enable it, pass the `--with-batch-norm` flag (and do let us know what we're doing wrong).

Example invocation:

```
python -m tf2qrnn.examples.classifier \
          --train-tsv examples_data/sent200.tsv \
          --data-column-name sentence \
          --gold-column-name sentiment \
          --label-map examples_data/document_classification_labels.txt \
          --model-weights-cp keras_weights/enwiki100_20epochs_35k_uncased \
          --model-type from_cp \
          --spm-model-file spm_models/enwiki100-uncased-sp35k.model \
          --fixed-seq-len 70 \
          --num-epochs 10 \
          --batch-size 16 \
          --lr 0.0025 \
          --lr-scheduler stlr \
          --out-path ./sent200trained \
          --qrnn --awd-off \
          --save-best
```

Now your classifier is ready in the `sent200trained` directory. The above command trains a classifier on a toy dataset and is almost guaranteed to overfit, but do give it a try with a demo:

```
python -m tf2qrnn.examples.classifier \
          --label-map examples_data/document_classification_labels.txt \
          --model-weights-cp sent200trained/best_checkpoint/best \
          --qrnn \
          --model-type from_cp \
          --spm-model-file enwiki100-uncased-sp35k.model \
          --fixed-seq-len 70 \
          --interactive
          

Paste a document to classify: this is the most depressing film i've ever seen . so boring i left before it finished .
[('POSITIVE', 0.08279895782470703), ('NEGATIVE', 0.917201042175293)]
Classification result: P(NEGATIVE) = 0.8749799728393555
Paste a document to classify: this is the most fascinating film i've ever seen . so captivating i wish it went for another hour .
[('POSITIVE', 0.998953104019165), ('NEGATIVE', 0.0010468183318153024)]
Classification result: P(POSITIVE) = 0.998953104019165
```



### 4.4. Regressor (`regressor.py`)

Sometimes instead of predicting a label from a fixed set of classes, you may want to predict a number. For instance, instead of classifying a hotel review as 'positive', 'negative', or 'neutral', you may want to predict the number of stars it was given. This is a regression task and and it turns out we can use ULMFiT for that purpose as well.

We have observed that the regressor training often goes terribly wrong if the dataset is unbalanced. If there are some values that tend to dominate (e.g. reviews with 5 stars), the network will learn to output values between 4 and 6 regardless of input. If the imbalance isn't very large, you may do well with oversampling or adding weights to the loss function, but only up to the point where the data is better handled using anomaly detection techniques.

Example invocation:

```
python -m tf2qrnn.examples.regressor \
          --train-tsv examples_data/hotels200.tsv \
          --data-column-name review \
          --gold-column-name rating \
          --model-weights-cp keras_weights/enwiki100_20epochs_35k_uncased \
          --model-type from_cp \
          --spm-model-file spm_model/enwiki100-uncased-sp35k.model \
          --fixed-seq-len 70 \
          --batch-size 16 \
          --num-epochs 12 \
          --lr 0.0025 \
          --lr-scheduler stlr \
          --loss-fn mse \
          --qrnn --awd-off \
          --out-path ./hotel_regressor \
          --save-best
```

The `--loss-fn` can be set to `mse` (mean squared error) or `mae` (mean absolute error). You can also pass `--normalize-labels` to squeeze the response values into the range between 0 and 1, which often improves the optimizer's convergence.

After training the regressor, run the demo:

```
python -m tf2qrnn.examples.regressor \
          --model-weights-cp hotel_regressor/best_checkpoint/best \
          --model-type from_cp \
          --qrnn \
          --spm-model-file spm_model/enwiki100-toks-sp35k-uncased.model \
          --fixed-seq-len 70 \
          --interactive

...

Paste a document to classify using a regressor: horrible hotel no security stole laptop avoid like plague
Score: = [2.198639]
Paste a document to classify using a regressor: excellent hotel stay wonderful staff
Score: = [5.979812]
```



Compare this to a model trained with `--normalize-labels` (note: this does not guarantee that the actual output will be in the range from 0 to 1, as illustrated by the second example)

```
Paste a document to classify using a regressor: horrible hotel no security stole laptop avoid like plague
Score: = [0.00999162]
Paste a document to classify using a regressor: excellent hotel stay wonderful staff
Score: = [1.145164]
```



### 4.5. Sequence tagger (`sequence_tagger.py`)

A sequence tagging task involves marking each token as belonging to one of the classes. The typical examples here are Named Entity Recognition (NER) and Part-of-Speech (POS) tagging. Our example script uses input data with annotations on the word level (a small sample of sentences from the CONLL2013 dataset). These are provided as a JSONL file containing lists of tuples:

```
[["Daniel", 5], ["Westlake", 6], [",", 0], ["46", 0], [",", 0], ["from", 0], ["Victoria", 1], [",", 0], ["made", 0], ["the", 0], ["first", 0], ["sucessful", 0], ["escape", 0], ["from", 0], ["Klongprem", 1], ["prison", 0], ["in", 0], ["the", 0], ["northern", 0], ["outskirts", 0], ["of", 0], ["the", 0], ["capital", 0], ["on", 0], ["Sunday", 0], ["night", 0], [".", 0]]
```

The numbers in the second element of each tuple are the token label defined in the labels map JSON file:

```
{"0": "O", "1": "B-LOC", "2": "I-LOC", "3": "B-ORG",
 "4": "I-ORG", "5": "B-PER", "6": "I-PER"}
```

The vast majority of publicly available datasets for such tasks contain annotations on the word level. However, our version of MultiFiT / ULMFiT uses subword tokenization which is almost universally adopted in newer language models. This means that the labels need to be re-aligned to subwords. The `sequence_tagger.py` script contains a simple function called `tokenize_and_align_labels` which does that for you. 

* Input word-level annotations from the training corpus

  ```
  Daniel         B-PER
  Westlake       I-PER
  ,                 O
  46                O
  from              O
  Victoria       B-LOC
  ...
  ```

* Annotations re-aligned to subwords:

  ```
  <s>               O
  ▁Daniel        B-PER
  ▁West          I-PER
  lake           I-PER
  ▁,                O
  ▁46               O
  ▁,                O
  ▁from             O
  ▁Victoria      B-LOC
  ...
  </s>              O
  ```


Example invocation (training - note: you need the *cased* model):

```
python -m tf2qrnn.examples.sequence_tagger \
          --train-jsonl examples_data/conll1000.jsonl \
          --label-map examples_data/tagger_labels.json \
          --spm-model-file spm_model/enwiki100-toks-sp35k-cased.model \
          --model-weights-cp keras_weights/enwiki100_20epochs_toks_35k_cased \
          --model-type from_cp \
          --batch-size 32 \
          --num-epochs 5 \
          --fixed-seq-len 70 \
          --save-best \
          --qrnn --awd-off \
          --out-path ./conll_tagger
```

Example invocation (demo - note the side effect of coarse alignment of B-LOC with all subwords of the first word in the training corpus. A more careful approach would have been to align B-LOC with only the first subword. 'Melfordshire' could then be tagged as `_M/B-LOC, elf/I-LOC, ord/I-LOC, shire/I-LOC`):

```
python -m tf2qrnn.examples.sequence_tagger \
          --label-map examples_data/tagger_labels.json \
          --model-weights-cp ./conll_tagger/tagger_best/best \
          --model-type from_cp \
          --spm-model-file spm_model/enwiki100-toks-sp35k-cased.model \
          --qrnn \
          --fixed-seq-len 70 \
          --interactive

Write a sentence to tag: What is the weather like in England ?
<s>               O
▁What             O
'                 O
s                 O
▁the              O
▁weather          O
▁like             O
▁in               O
▁England       B-LOC
▁                 O
?                 O
</s>              O
<pad>             O
<pad>             O
<pad>             O
.......
```





## 5. Pretraining your own language model from scratch (`pretraining_utils`)

This section describes how to use a raw text corpus to train an ULMFiT or MultiFiT language model. As explained at the beginning of this document, we use FastAI to train encoder weights, which we then convert to a Tensorflow model with the [convert_fastai2keras.py](convert_fastai2keras.py) script.

Obtaining source data and cleaning it up will require different techniques for each data source. Here we assume that you are already past this stage and **that you already have a reasonably clean raw corpus**. All you need to do is save it as three large plain text files (e.g. `train.txt`, `valid.txt` and `test.txt`) with one sentence per line. This is important since our scripts add BOS and EOS markers at the beginning and end of each line. As an alternative, you may want to train a language model on paragraphs or documents - in which each line in your text files will need to correspond to a paragraph or a document.



### 5.1. Basic cleanup and sentence tokenization (optional - `01_cleanup.py`)

If your text isn't already sentence-tokenized, you can split it into sentences in separate lines with the `01_cleanup.py` script. Some minor cleanup and preprocessing is also performed, in particular punctuation characters are separated from words by a whitespace (ex. `what?` -> `what ?`).

Example invocation:

```
python -m tf2qrnn.pretraining_utils.01_cleanup \
          --lang english \
          --input-text train_raw.txt \
          --out-path ./train_sents.txt
```



### 5.2. Build subword vocabulary (`02_build_spm.py`)

To avoid problems with out-of-vocabulary words, we recommend using a subword tokenizer such as **[Sentencepiece](https://github.com/google/sentencepiece)** (SPM). An important decision to take is how many subwords you want to have. Unless you are building something as big as GPT-3, it's probably safe to use ~30k subwords for English and maybe up to 50k for highly inflectional languages.

Those of you who are proficient in using sentencepiece from the command line can build SPM from Google's repo and then run `spm_train` manually. The advantage is that you get to tweak things like character coverage, but remember to set the control character indices as expected by FastAI (see section 3.1) and not to their default values. If you don't need this much flexibility, you can run our wrapper around SPM trainer in `02_build_spm.py`.

Example invocation:

```
python -m tf2qrnn.pretraining_utils.02_build_spm \
          --corpus-path train_sents.txt \
          --vocab-size 5000 \
          --model-prefix wittgenstein
```

This will produce two files: `wittgenstein-sp5k.model` and `wittgenstein-sp5k.vocab`. You really only need the first one for further steps.



### 5.3. Tokenize and numericalize your corpus (`03_encode_spm.py`) 

Now that you have the raw text and the sentencepiece tokenizer, you can convert text to tokens and their IDs:

- **Tokenization:** (observe the BOS/EOS markers!)
  
  ```
  ['<s>', '▁The', '▁lim', 'its', '▁of', '▁my', '▁language', '▁mean', '▁the', '▁lim', 'its', '▁of', '▁my', '▁world', '</s>']
  ```
  
- **Numericalization** (converting each subword to its ID in the dictionary):
  
  ```
  [2, 200, 2240, 1001, 20, 317, 2760, 286, 9, 2240, 1001, 20, 317, 669, 3]
  ```
  
  

Because it was important for us to have full control over numericalization (we didn't want to use FastAI's default functions), our training script requires the corpus to be numericalized before being fed to FastAI data loaders. The `03_encode_spm.py` file will read a plain text file with the source corpus and save another text file with its numericalized version. It will also add BOS (`2`) and EOS (`3`) markers.

Example invocation:

```
python -m tf2qrnn.pretraining_utils.03_encode_spm \
          --corpus-path ./train_sents.txt \
          --model-path ./wittgenstein-sp5k.model \
          --spm-extra-options bos:eos \
          --output-format id \
          --save-path ./train_ids.txt
```



### 5.4. Training in FastAI (`fastai_ulmfit_train.py`)

Congratulations! Now that your data is mangled, all you need to do is execute FastAI training. **Only do it on a CPU if your corpus is really tiny** as pretraining takes a significant amount of time. Before you run the training we advise that you study the argument descriptions (`--help` flag ) carefully, as there are several hyperparameters that will for sure need tweaking on your custom input. For large datasets (e.g. Wikipedia) in particular it will be useful to prepare a numericalized validation corpus and pass it via the `--pretokenized-valid` argument so that you can measure the perplexity metric after each epoch.

Example invocation:

```
python -m tf2qrnn.fastai_ulmfit_train \
       --pretokenized-train ./train_ids.txt \
       --min-seq-len 7 \
       --max-seq-len 80 \
       --batch-size 128 \
       --vocab-size 5000 \
       --num-epochs 20 \
       --save-path ./wittgenstein_lm \
       --exp-name wittg

...
epoch     train_loss  valid_loss  accuracy  perplexity  time    
0         6.454500    5.968120    0.094213  390.770355  02:40
1         5.841645    5.346557    0.181822  209.884445  02:35
2         5.404020    4.939891    0.204419  139.755066  03:07
3         5.085054    4.652817    0.228302  104.879990  03:35
4         4.877162    4.478850    0.241150  88.133247   03:38
....
```

The result is as FastAI model file called `wittg.pth` which will be saved in the `wittgenstein_lm` directory.



**5.5. Exporting to Tensorflow 2.0 Keras weights and SavedModel (`convert_fastai2keras.py`)**

Weights from the file created by FastAI should be exported to a format usable by Tensorflow. This is done with the `convert_fastai2keras.py` script:

```
CUDA_VISIBLE_DEVICES=-1 python -m tf2qrnn.convert_fastai2keras.py \
    --pretrained-model wittgenstein_lm/wittg.pth \
    --spm-model-file ./wittgenstein-sp5k.model \
    --out-path ./tf_wittgenstein
```

The output of the conversion script is a directory tree like this:

```
├── fastai_model
│   └── wittg.pth
├── keras_weights
│   ├── checkpoint
│   ├── wittg.data-00000-of-00001
│   └── wittg.index
├── saved_model
│   ├── assets
│   │   ├── wittgenstein-sp5k.model
│   │   └── wittgenstein-sp5k.vocab
│   ├── saved_model.pb
│   └── variables
│       ├── variables.data-00000-of-00001
│       └── variables.index
└── spm_model
    ├── wittgenstein-sp5k.model
    └── wittgenstein-sp5k.vocab
```

As you can see it contains four subdirectories:

* **fastai_model** - the original file used by FastAI. You still need that .pth file if you want to do unsupervised finetuning on a target (unlabelled) domain data because our Tensorflow implementatiion does not support that yet.

* **keras_weights** - the checkpoint which you can restore into a Keras model created from python code

* **saved_model** - a serialized version of the model stored together with its weights. This will produce modules with three signatures: `string_encoder`, `numericalized_encoder` and `spm_processor` (see the section on restoring from a SavedModel above).



## 6. References and acknowledgements

* [Universal Language Model Fine-tuning for Text Classification](https://arxiv.org/abs/1801.06146) - the original paper on improvements made to the LSTM architecture
* [MultiFiT: Efficient Multi-lingual Language Model Fine-tuning](https://aclanthology.org/D19-1572.pdf) - the paper describing the model in which LSTM cells were replaced with QRNN ones
* [Transfer learning in text](https://docs.fast.ai/tutorial.text.html#The-ULMFiT-approach) - part of FastAI docs
* [Universal Language Model Fine-Tuning (ULMFiT): State-of-the-Art in Text Analysis](https://humboldt-wi.github.io/blog/research/information_systems_1819/group4_ulmfit/#ttc) - analysis by researchers from the Humboldt University in Berlin
* [Understanding building blocks of ULMFiT](https://blog.mlreview.com/understanding-building-blocks-of-ulmfit-818d3775325b) - blog post with detailed description and examples of dropouts used in the ULMFiT model

The code in this repo was forked from my previous work on porting ULMFiT to Tensorflow and [published here](https://bitbucket.org/edroneteam/tf2_ulmfit/src/master/). Many thanks to my employer, [edrone](https://edrone.me/), for providing GPU machines for training.

Contact me at [hk@hubertkarbowy.pl](hk@hubertkarbowy.pl) if you find bugs or need support.
