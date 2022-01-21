# note: this code is copied from https://github.com/hubertkarbowy/tf2_qrnn

import tensorflow as tf

class QRNNCell_(tf.keras.layers.Layer):
    """
    A QRNN cell usable with Keras' RNN API in TF 2.0.

    This is a drop-down replacement for tf.keras.layers.LSTMCell (or SimpleRNNCell, or GRUCell) and can
    be plugged into any existing code without adaptations.

    Statefulness and return_sequences is handled via Keras RNN API - no need to write these explicitly here.

    Not optimized for CUDA, not tested when serializing/deserializing into a SavedModel - contributions are
    welcome. Leave a pull request or send me an email: hk*at_symbol^hubertkarbowy.pl.
    """


    def __init__(self, hidden_units, *, stateful=False, kernel_window=2, zoneout=0.0,
                 pooling='fo', return_state=False, return_sequences=False, name=None,
                 keep_fastai_bug=True, **kwargs):
        super().__init__(name=name, **kwargs)
        if kernel_window < 2:
            # raise ValueError("The convolutional filter's width must be greater than 1.")
            print("Kernel windows of size 1 kind of don't make sense with QRNNs...")
        self.hidden_units = hidden_units
        self.stateful = stateful
        self.kernel_window = kernel_window
        self.zoneout = zoneout
        self.pooling = pooling
        self.return_state = return_state
        self.return_sequences = return_sequences
        if pooling == 'f':
            self.num_matrices_in_fused_kernel = 2
        elif pooling == 'fo':
            self.num_matrices_in_fused_kernel = 3
        elif pooling == 'ifo':
            self.num_matrices_in_fused_kernel = 4
        else:
            raise ValueError(f"Unknown pooling method {pooling}. Available variants are: 'f', 'fo' or 'ifo'.")
        self.num_input_features = None
        ### these attributes are mandated by the Keras RNN API: ###
        self.state_size = None # e.g. [64, 70]  - the number of hidden units in each state vector. As zeroeth element
                               # we pass the memory (c_t), as first element we pass inputs from x_{t-window_size+1}
                               # to x_t inclusive. Please make sure you understand the reasons for this.
        self.output_size = hidden_units  # e.g. 64 - output (h_t) when using fo-pooling and ifo-pooling,
                                         # or simply c_t when using f-pooling.
        self.keep_fastai_bug = keep_fastai_bug


    def build(self, input_shape): # input shape is (batch_size, hidden_units), not (batch_size, num_steps, hidden_units)
        print(f"Entering build with input shape {input_shape}")
        if self.keep_fastai_bug:
            tf.print("Warning: this version of a QRNNCell is numerically compatible with `forget_mult_CPU` code from fastai, " \
                     "which computes `c_t`, but after close scrutiny and comparison with the QRNN paper, it seems fastai's " \
                     "implementation of Equation (4) is incorect. Also, they pass the `h_t` to the next time step as the" \
                     "cell's memory, and output `c_t` as the representation of sequence up-to-and-including the current time step, "\
                     "not vice versa as was in an LSTM network and as recurrence in Eq (4) suggests.\n\nIf you have imported weights from " \
                     "fastai, you will get identical results, but do bear in mind that their implementation looks suspicious")

        self.num_input_features = input_shape[1] # tu trzeba wydobyc rozmiar z poprzedniej warstwy (np. 400 dla embeddingow)
        self.state_size = [self.hidden_units, self.num_input_features * (self.kernel_window - 1)]
        weights_shape = (self.num_input_features * self.kernel_window, self.num_matrices_in_fused_kernel * self.hidden_units)
        self.fused_weights = self.add_weight(shape=weights_shape,
                                              initializer='uniform',
                                              name='fused_gate_weights',
                                              trainable=True)
        self.fused_biases  = self.add_weight(shape=(self.num_matrices_in_fused_kernel * self.hidden_units),
                                              initializer='zeros',
                                              name='fused_gate_biases',
                                              trainable=False)
        self.built = True


    def get_initial_state(self, inputs=None, batch_size=None, dtype=None):
        """
        The returned initial state should have a shape of [batch_size, cell.state_size]
        """
        if inputs is not None:
            batch_size = tf.shape(inputs)[0]
            num_input_features = tf.shape(inputs)[1]
            dim = self.hidden_units
            dtype = inputs.dtype
            # the second element will repeat the inputs for the convenience of convolutions
            return [tf.zeros((batch_size, dim)), tf.zeros((batch_size, num_input_features*(self.kernel_window -1)))]
        else:
            return [tf.zeros((batch_size, self.hidden_units)), tf.zeros((batch_size, self.num_input_features*(self.kernel_window - 1)))]
            

    def call(self, inputs, states=None, training=None):
        num_input_features = tf.shape(inputs)[1]
        print(num_input_features)
        incoming_memory = states or self.get_initial_state(inputs)
        inputs_at_t = inputs
        inputs_at_t_minus_k = incoming_memory[1]
        c_t_minus_1 = incoming_memory[0]
        # The line below does the same thing as a rather verbose `_get_source(inp)` fuction in fastai code.
        # It also ensures "masked convolutions" at the zeroeth time step. See the readme file to understand
        # how the inputs from previous time steps are obtained at the present time step.
        merged_timesteps = tf.concat([inputs_at_t, inputs_at_t_minus_k], axis=1)
        fused_logits = tf.matmul(merged_timesteps, self.fused_weights) + self.fused_biases
        # The verbosity / redundancy in the ifology below is intentional - it will generate fewer
        # problems during graph tracing and serialization.
        if self.pooling == 'f':
            z_gate, f_gate = tf.split(fused_logits, self.num_matrices_in_fused_kernel, axis=1)
            z_gate = tf.tanh(z_gate)
            f_gate = tf.sigmoid(f_gate)
            if training:
                f_gate = 1 - tf.nn.dropout(1 - f_gate, rate=self.zoneout)
            if self.keep_fastai_bug:  #  according to fastai's (broken?) `forget_mult_CPU`:
                c_t = (tf.multiply(z_gate, f_gate)) + tf.multiply((1 - f_gate), c_t_minus_1)
            else:  #  according to the QRNN paper:
                c_t = (tf.multiply(f_gate, c_t_minus_1)) + tf.multiply((1 - f_gate), z_gate)    
            h_t = c_t
        elif self.pooling == 'fo':
            z_gate, f_gate, o_gate = tf.split(fused_logits, self.num_matrices_in_fused_kernel, axis=1)
            f_gate = tf.sigmoid(f_gate)
            z_gate = tf.tanh(z_gate)
            o_gate = tf.sigmoid(o_gate)
            if training:
                f_gate = 1 - tf.nn.dropout(1 - f_gate, rate=self.zoneout)
            if self.keep_fastai_bug:  #  according to fastai's (broken?) `forget_mult_CPU`:
                h_t = (tf.multiply(z_gate, f_gate)) + tf.multiply((1 - f_gate), c_t_minus_1)
                c_t = tf.multiply(o_gate, h_t)
            else:  #  according to the QRNN paper:
                c_t = tf.multiply(f_gate, c_t_minus_1) + tf.multiply(1 - f_gate, z_gate)
                h_t = tf.multiply(o_gate, c_t)
        elif self.pooling == 'ifo':
            z_gate, f_gate, o_gate, i_gate = tf.split(fused_logits, self.num_matrices_in_fused_kernel, axis=1)
            f_gate = tf.sigmoid(f_gate)
            z_gate = tf.tanh(z_gate)
            o_gate = tf.sigmoid(o_gate)
            i_gate = tf.sigmoid(i_gate)
            if training:
                f_gate = 1 - tf.nn.dropout(1 - f_gate, rate=self.zoneout)
            c_t = tf.multiply(f_gate, c_t_minus_1) + tf.multiply(i_gate, z_gate)
            h_t = tf.multiply(o_gate, c_t)
        
        relevant_past_timesteps = tf.slice(merged_timesteps, [0, 0], [-1, (self.kernel_window-1)*num_input_features])
        states_at_t_plus_1 = [h_t, relevant_past_timesteps] # we pass the inputs from t to t+1 to allow convolutions
        return [c_t, states_at_t_plus_1]
