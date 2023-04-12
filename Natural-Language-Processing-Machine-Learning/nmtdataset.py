"""Build Dataset for this project."""


import io
import re
import unicodedata

import tensorflow as tf
from sklearn.model_selection import train_test_split


class NMTDataset:
    """Build Dataset for this project.

    Depending on the problem_type, the dataset will be created.
    It is possible to check if the dataset already exists.
    If not, it will be downloaded from https://www.nmt.org/data.
    Note: the downloading process will not be introduced here.
    """

    def __init__(self, problem_type="en-vie", path=""):
        self.problem_type = problem_type
        self.inp_lang_tokenizer = None
        self.targ_lang_tokenizer = None
        self.path = path

    def unicode_to_ascii(self, s):
        return "".join(
            c
            for c in unicodedata.normalize("NFD", s)
            if unicodedata.category(c) != "Mn"
        )

    ## Step 1 and Step 2
    def preprocess_sentence(self, w):
        w = self.unicode_to_ascii(w.lower().strip())

        # creating a space between a word and the punctuation following it
        # eg: "he is a boy." => "he is a boy ."
        # Reference:-
        # https://stackoverflow.com/questions/3645931/python-padding-punctuation-with-white-spaces-keeping-punctuation
        w = re.sub(r"([?.!,¿])", r" \1 ", w)
        w = re.sub(r'[" "]+', " ", w)

        # replacing everything with space except (a-z, A-Z, ".", "?", "!", ",")
        w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

        w = w.strip()

        # adding a start and an end token to the sentence
        # so that the model know when to start and stop predicting.
        w = "<start> " + w + " <end>"
        return w

    def create_dataset(self, path, num_examples):
        if not path:
            path = self.path
        # path : path to spa-eng.txt file
        # num_examples : Limit the total number of training example for
        #   faster training (set num_examples = len(lines) to use full data)
        lines = io.open(path, encoding="UTF-8").read().strip().split("\n")
        word_pairs = [
            [self.preprocess_sentence(w) for w in l.split("\t")]
            for l in lines[:num_examples]
        ]

        return zip(*word_pairs)

    # Step 3 and Step 4
    def tokenize(self, lang):
        # lang = list of sentences in a language

        # print(len(lang), "example sentence: {}".format(lang[0]))
        lang_tokenizer = tf.keras.preprocessing.text.Tokenizer(
            filters="", oov_token="<OOV>"
        )
        lang_tokenizer.fit_on_texts(lang)

        # tf.keras.preprocessing.text.Tokenizer.texts_to_sequences converts
        # string (w1, w2, w3, ......, wn) to a list of correspoding integer ids
        # of words (id_w1, id_w2, id_w3, ...., id_wn)
        tensor = lang_tokenizer.texts_to_sequences(lang)

        # tf.keras.preprocessing.sequence.pad_sequences takes argument a list
        # of integer id sequences and pads the sequences to match the longest
        # sequences in the given input
        tensor = tf.keras.preprocessing.sequence.pad_sequences(
            tensor, padding="post"
        )

        return tensor, lang_tokenizer

    def load_dataset(self, path, num_examples=None):
        if not path:
            path = self.path
        # creating cleaned input, output pairs
        targ_lang, inp_lang = self.create_dataset(path, num_examples)

        input_tensor, inp_lang_tokenizer = self.tokenize(inp_lang)
        target_tensor, targ_lang_tokenizer = self.tokenize(targ_lang)

        return (
            input_tensor,
            target_tensor,
            inp_lang_tokenizer,
            targ_lang_tokenizer,
        )

    def call(self, num_examples, BUFFER_SIZE, BATCH_SIZE):
        file_path = ""  # get_nmt()
        (
            input_tensor,
            target_tensor,
            self.inp_lang_tokenizer,
            self.targ_lang_tokenizer,
        ) = self.load_dataset(file_path, num_examples)

        (
            input_tensor_train,
            input_tensor_val,
            target_tensor_train,
            target_tensor_val,
        ) = train_test_split(input_tensor, target_tensor, test_size=0.2)

        train_dataset = tf.data.Dataset.from_tensor_slices(
            (input_tensor_train, target_tensor_train)
        )
        train_dataset = train_dataset.shuffle(BUFFER_SIZE).batch(
            BATCH_SIZE, drop_remainder=True
        )

        val_dataset = tf.data.Dataset.from_tensor_slices(
            (input_tensor_val, target_tensor_val)
        )
        val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)

        return (
            train_dataset,
            val_dataset,
            self.inp_lang_tokenizer,
            self.targ_lang_tokenizer,
        )
