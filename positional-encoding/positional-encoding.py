import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    
    
    # Write code here
    
    
    
    import numpy as np

def positional_encoding(seq_len, d_model, base=10000.0):
    """
    Return PE of shape (seq_len, d_model) using sin/cos formulation.
    Odd d_model -> last column is sin.
    """
    # positions: (seq_len, 1)
    positions = np.arange(seq_len)[:, np.newaxis]
    
    # dimensions: (1, d_model)
    dims = np.arange(d_model)[np.newaxis, :]
    
    # compute the denominator term
    angle_rates = 1 / (base ** (2 * (dims // 2) / d_model))
    
    # compute angle matrix
    angle_rads = positions * angle_rates
    
    # initialize PE
    pe = np.zeros((seq_len, d_model))
    
    # apply sin to even indices (0,2,4,...)
    pe[:, 0::2] = np.sin(angle_rads[:, 0::2])
    
    # apply cos to odd indices (1,3,5,...)
    pe[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    return pe
    pass


    