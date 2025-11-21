import random

import torch
from tqdm import tqdm
import numpy as np
from hftokenizer import HFTokenizer
import warnings
warnings.filterwarnings("ignore", message="Token indices sequence length is longer")


def construct_dataset(data_txt_file, sequence_length=256):
    """
    data_txt_file : a string path to a text file containing training data, one sample per line
    sequence_length : int, the desired length of each training sequence

    This method should use the trained tokenizer to convert samples to token_ids, and
    then pack them into a training set represented as a 2D array of size (sequences, sequence_length+1).
    The +1 is very important! It lets us compare our model outputs to the sequence shifted by one.

    You can save this training set in whatever format you wish for loading into the training script.
    I recommend using numpy's np.save() method or the pickle module.

    The saved data should be shuffled so we can directly load it and train on it in the training script.
    """

    # construct tokenizer
    tokenizer = HFTokenizer()
    tokenizer.load()

    # get all samples
    f = open(data_txt_file, "r", encoding='utf-8')
    samples = f.readlines()
    samples = [x.replace("\n", "") for x in samples]

    # ----------------------------------------
    # - add '<|endoftext|>' to each sample or add tokenizer.eos_token_id after tokenizing.
    # - use tokenizer.encode() to tokenize each sample
    print("Encoding samples")
    encoded = [tokenizer.tokenizer.encode(sample + tokenizer.tokenizer.eos_token) for sample in tqdm(samples)]
    # - pack into sequences of length sequence_length
    print("\nPacking sequences")
    packed_sequences = _pack_sequences(sequence_length, encoded)
    # - shuffle
    print("\nShuffling sequences")
    shuffled_sequences = _shuffle_sequences(packed_sequences)
    # - save out data
    print("\nSaving shuffled tokens")
    np.save('shuffled_tokens_old.npy', shuffled_sequences)
    print("\nFinished constructing the dataset")


def _pack_sequences(sequence_length, encoded):
    """
    Packs the encodings into sequences of the same length. If the last sequence does not have enough tokens
    to fill the sequence length, it is thrown out.

    :param sequence_length: The sequence length all the packed sequences should be.
    :param encoded: The list of encoded samples (each ending with an eos token)
    :return: The list of packed sequences
    """
    packed_sequences = []  # place an empty list as the first element of the sequences
    current_sequence = []
    for e in tqdm(encoded):
        for token in e:
            current_sequence.append(token)
            if len(current_sequence) == sequence_length + 1: # use sequence_length + 1, so we can predict later
                packed_sequences.append(current_sequence)
                current_sequence = [token]  # the token should be the start of the new sequence as well
    # if the last sequence doesn't have enough tokens, remove it
    if len(packed_sequences[-1]) < sequence_length + 1:
        packed_sequences = packed_sequences[:-1]
    return packed_sequences


def _shuffle_sequences(sequences):
    """
    Shuffles the packed sequences. Each sequence is treated as its own entity and the elements within that
    individual sequence are shuffled.
    """
    # shuffled_sequences = []
    # for sequence in tqdm(sequences):
    #     sequence = sequence[:]
    #     random.shuffle(sequence)
    #     shuffled_sequences.append(sequence)
    random.shuffle(sequences)
    return np.array(sequences, dtype=np.int64)


if __name__ == "__main__":
    construct_dataset("./data.txt", 256)

"""
References
    Lecture notes for module 6
    Documentation from https://github.com/tqdm/tqdm because I had never used this library before!
    Very cool and easy to use. I will definitely be adding progress bars to my future projects.
    https://huggingface.co/docs/transformers/en/pad_truncation -> documentation for encoding
"""