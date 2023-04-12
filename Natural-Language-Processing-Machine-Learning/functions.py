"""Functions for building NLP models."""

import tensorflow as tf
import tensorflow_addons as tfa


def evaluate_sentence(
    sentence, inp_lang, targ_lang, encoder, decoder, max_length_input, units
):
    """Evaluate a single sentence using tf-addons BasicDecoder.

    The function has many arguments, but only one is required. The others were
    adapted to separate the code from main file. If the code of this function
    on the main file, remove all arguments but the first one.
    More information about tf-addons BasicDecoder can be found here:
    https://www.tensorflow.org/addons/tutorials/networks_seq2seq_nmt

    Parameters
    ----------
    sentence : str
        An input sentence.

    Returns
    -------
    string : str
        The translated sentence.
    """
    inputs = [inp_lang.word_index[i] for i in sentence.split(" ")]
    inputs = tf.keras.preprocessing.sequence.pad_sequences(
        [inputs], maxlen=max_length_input, padding="post"
    )
    inputs = tf.convert_to_tensor(inputs)
    inference_batch_size = inputs.shape[0]

    enc_start_state = [
        tf.zeros((inference_batch_size, units)),
        tf.zeros((inference_batch_size, units)),
    ]
    enc_out, enc_h, enc_c = encoder(inputs, enc_start_state)

    start_tokens = tf.fill(
        [inference_batch_size], targ_lang.word_index["<start>"]
    )
    end_token = targ_lang.word_index["<end>"]

    greedy_sampler = tfa.seq2seq.GreedyEmbeddingSampler()

    # Instantiate BasicDecoder object
    decoder_instance = tfa.seq2seq.BasicDecoder(
        cell=decoder.rnn_cell, sampler=greedy_sampler, output_layer=decoder.fc
    )
    # Setup Memory in decoder stack
    decoder.attention_mechanism.setup_memory(enc_out)

    # set decoder_initial_state
    decoder_initial_state = decoder.build_initial_state(
        inference_batch_size, [enc_h, enc_c], tf.float32
    )

    # Since the BasicDecoder wraps around Decoder's rnn cell only, you have to
    # ensure that the inputs to BasicDecoder decoding step is output of
    # embedding layer. tfa.seq2seq.GreedyEmbeddingSampler() takes care of this.
    # You only need to get the weights of embedding layer, which can be done
    # by decoder.embedding.variables[0] and pass this callabble to
    # BasicDecoder's call() function

    decoder_embedding_matrix = decoder.embedding.variables[0]

    outputs, _, _ = decoder_instance(
        decoder_embedding_matrix,
        start_tokens=start_tokens,
        end_token=end_token,
        initial_state=decoder_initial_state,
    )
    return outputs.sample_id.numpy()


optimizer = tf.keras.optimizers.Adam()


def loss_function(real, pred):
    """Customize the loss function for Training steps.

    Parameters
    ----------
    real : array-like of sentences
        An array of input: ground truth.
        real shape = (BATCH_SIZE, max_length_output)
    pred : array-like of sentences
        Result of the model needed to compare with ground truth.
        pred shape = (BATCH_SIZE, max_length_output, tar_vocab_size )

    Returns
    -------
    loss : float
        In this case, the loss function is SparseCategoricalCrossentropy.
    """
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    loss = cross_entropy(y_true=real, y_pred=pred)
    mask = tf.logical_not(tf.math.equal(real, 0))  # output=0 for y=0 else 1
    mask = tf.cast(mask, dtype=loss.dtype)
    loss = mask * loss
    loss = tf.reduce_mean(loss)
    return loss


@tf.function
def train_step(inp, targ, enc_hidden, BATCH_SIZE, encoder, decoder):
    """Train the model on a single training step for each batch of sentences.

    We customize the training step to feed to the Encoder, followed by
    the Decoder.
    """
    loss = 0

    with tf.GradientTape() as tape:
        enc_output, enc_h, enc_c = encoder(inp, enc_hidden)

        dec_input = targ[:, :-1]  # Ignore <end> token
        real = targ[:, 1:]  # Ignore <start> token

        # Set the AttentionMechanism object with encoder_outputs
        decoder.attention_mechanism.setup_memory(enc_output)

        # Create AttentionWrapperState as initial_state for decoder
        decoder_initial_state = decoder.build_initial_state(
            BATCH_SIZE, [enc_h, enc_c], tf.float32
        )
        pred = decoder(dec_input, decoder_initial_state)
        logits = pred.rnn_output
        loss = loss_function(real, logits)

    variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))

    return loss
