# Quasi-Recurrent Neural Network (QRNN) for TensorFlow 2.0

###Intro
This is a drop-down implementation. Finally in 2022 we have this working QRNN code that does not rely  on some antique TF version and is usable in Keras just as easily as any other layer.


###Quick start

Replacing an LSTM / GRU cell with QRNN has never been easier. Here is a simple example:

```
import tensorflow as tf
from qrnn_tf2 import QRNNCell
model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=(60,), dtype="int32"))
model.add(tf.keras.layers.Embedding(5000, 128))
model.add(tf.keras.layers.RNN(QRNNCell(128, zoneout=0.3), return_sequences=True))
model.add(tf.keras.layers.RNN(QRNNCell(64)))
model.add(tf.keras.layers.Dense(3))
model.summary()

Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding_1 (Embedding)      (None, 60, 128)           640000
_________________________________________________________________
rnn (RNN)                    (None, 60, 128)           98688
_________________________________________________________________
rnn_1 (RNN)                  (None, 64)                49344
_________________________________________________________________
dense (Dense)                (None, 3)                 195
=================================================================
Total params: 788,227
Trainable params: 787,651
Non-trainable params: 576
_________________________________________________________________
```
By default there are three gates (fo-pooling), but you can instantiate other variants (see details below). Also, note that QRNN does not have recurrent weights and is regularized differently than other recurrent cells. There is no dropout or recurrent dropout - `zoneout` is used instead.


###Arguments
* `hidden_units`
* `kernel_window`
* `zoneout`
* `pooling`

The arguments `name`, `stateful`, `return_sequences` and `return_state` are handled by Keras API and work like a charm without having to implement them in this code.

**TO CHECK**: does masking work?


###Usage with `return_sequences` in the topmost-layer



IMPORTANT OBSCURE TRICK: we use the cell's states to pass inputs from previous time steps
to next steps in the sequence, in addition to the actual cell state!!


    Perform left-padding of the `inputs`.


    `inputs_at_t`          - (bsize, dim) for the current time step
    `inputs_at_t_minus_1`  - (bsize, dim) for the previous time step
    
    Given a sequence of `n` time steps, each represented by a vector of size `dim`,
    this function outputs a sequence of the same length consummable by a convolutional
    filter of size (window) `window_size`.
    
    The output vector arises from:
        1. copying the input sequence `window_size` times,
        2. shifting subsequent copies one time step to the right,
        3. padding the values on the left with zeros
        4. stacking all copies on top of each other.
    
    In this way we obtain left-padded sequences packed into a single vector for efficiency reasons.
    Notice that the number of time steps remains `n`, but each now has `dim` * `window_size` features.
    By performing left-padding (now listen very carefully, this is super important, and I shall
    say this only once!) at each time step `t`, a convolutional filter which consumes the output vector
    operates ONLY ON THE PAST INFORMATION, i.e. time steps up to and including `t` - `window_size` + 1.
    It DOES NOT HAVE ACCESS TO FUTURE INFORMATION, i.e. time steps after `t`. In computer vision this
    is known as a "masked convolution" (van den Oord 2016).

    The code that follows is basically a reimplementation of FastAI. Let's take a toy example to illustrate it -
    a batch of 2 tensors, each containing 4 time steps, each containing 3 features:
    
        <tf.Tensor: shape=(2, 4, 3), dtype=float32, numpy=
        array([[[  1.,   2.,   3.],
                [  4.,   5.,   6.],
                [  7.,   8.,   9.],
                [ 10.,  11.,  12.]],
    
               [[100., 110., 120.],
                [200., 210., 220.],
                [300., 310., 320.],
                [400., 410., 420.]]], dtype=float32)>
    
    After left-padding we have the following output:
    
        <tf.Tensor: shape=(2, 4, 6), dtype=float32, numpy=
        array([[[  1.,   2.,   3.,   0.,   0.,   0.],
                [  4.,   5.,   6.,   1.,   2.,   3.],
                [  7.,   8.,   9.,   4.,   5.,   6.],
                [ 10.,  11.,  12.,   7.,   8.,   9.]],
        
                [[100., 110., 120.,   0.,   0.,   0.],
                [200., 210., 220., 100., 110., 120.],
                [300., 310., 320., 200., 210., 220.],
                [400., 410., 420., 300., 310., 320.]]], dtype=float32)>




"""
