import numpy as np

def pad_sequences(seqs, pad_value=0, max_len=None):

    if len(seqs) == 0:
        return np.zeros((0,0), dtype=int)

    # find max length if not given
    if max_len is None:
        max_len = max(len(seq) for seq in seqs)

    padded = []

    for seq in seqs:

        if len(seq) > max_len:
            new_seq = seq[:max_len]          # truncate

        else:
            pad_count = max_len - len(seq)
            new_seq = seq + [pad_value] * pad_count

        padded.append(new_seq)

    return np.array(padded, dtype=int)