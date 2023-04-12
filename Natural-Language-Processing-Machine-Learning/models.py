"""Models Encoder and Decoder for this project."""


import tensorflow as tf
import tensorflow_addons as tfa

"""Sequence to Sequence (seq2seq) Model: Encoder and Decoder."""


class Encoder(tf.keras.Model):
    """Encoder is a keras Model to build LSTM layer."""

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
        super(Encoder, self).__init__()
        self.batch_sz = batch_sz
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        # LSTM layer in Encoder
        self.lstm_layer = tf.keras.layers.LSTM(
            self.enc_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer="glorot_uniform",
        )

    def call(self, x, hidden):
        x = self.embedding(x)
        output, h, c = self.lstm_layer(x, initial_state=hidden)
        return output, h, c

    def initialize_hidden_state(self):
        return [
            tf.zeros((self.batch_sz, self.enc_units)),
            tf.zeros((self.batch_sz, self.enc_units)),
        ]


class Decoder(tf.keras.Model):
    """Decoder is a keras Model to build LSTM layer."""

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        dec_units,
        batch_sz,
        max_length_input,
        max_length_output,
        attention_type="luong",
    ):
        super(Decoder, self).__init__()
        self.batch_sz = batch_sz
        self.dec_units = dec_units
        self.attention_type = attention_type
        self.max_length_input = max_length_input
        self.max_length_output = max_length_output

        # Embedding Layer
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # Final Dense layer on which softmax will be applied
        self.fc = tf.keras.layers.Dense(vocab_size)

        # Define the fundamental cell for decoder recurrent structure
        self.decoder_rnn_cell = tf.keras.layers.LSTMCell(self.dec_units)

        # Sampler
        self.sampler = tfa.seq2seq.sampler.TrainingSampler()

        # Create attention mechanism with memory = None
        self.attention_mechanism = self.build_attention_mechanism(
            self.dec_units,
            None,
            self.batch_sz * [self.max_length_input],
            self.attention_type,
        )

        # Wrap attention mechanism with the fundamental rnn cell of decoder
        self.rnn_cell = self.build_rnn_cell(batch_sz)

        # Define the decoder with respect to fundamental rnn cell
        self.decoder = tfa.seq2seq.BasicDecoder(
            self.rnn_cell, sampler=self.sampler, output_layer=self.fc
        )

    def build_rnn_cell(self, batch_sz):
        rnn_cell = tfa.seq2seq.AttentionWrapper(
            self.decoder_rnn_cell,
            self.attention_mechanism,
            attention_layer_size=self.dec_units,
        )
        return rnn_cell

    def build_attention_mechanism(
        self, dec_units, memory, memory_sequence_length, attention_type="luong"
    ):
        """Build attention mechanism according to attention_type.

        Parameters
        ----------
        dec_units : int
            final dimension of attention outputs
        memory : _type_
            encoder hidden states of shape
            (batch_size, max_length_input, enc_units)
        memory_sequence_length : _type_
            1d array of shape (batch_size) with every element set to
            max_length_input(for masking purpose)
        attention_type : str, optional
            Which sort of attention (Bahdanau, Luong)
        """
        if attention_type == "bahdanau":
            return tfa.seq2seq.BahdanauAttention(
                units=dec_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
            )
        else:
            return tfa.seq2seq.LuongAttention(
                units=dec_units,
                memory=memory,
                memory_sequence_length=memory_sequence_length,
            )

    def build_initial_state(self, batch_sz, encoder_state, Dtype):
        decoder_initial_state = self.rnn_cell.get_initial_state(
            batch_size=batch_sz, dtype=Dtype
        )
        decoder_initial_state = decoder_initial_state.clone(
            cell_state=encoder_state
        )
        return decoder_initial_state

    def call(self, inputs, initial_state):
        x = self.embedding(inputs)
        outputs, _, _ = self.decoder(
            x,
            initial_state=initial_state,
            sequence_length=self.batch_sz * [self.max_length_output - 1],
        )
        return outputs
