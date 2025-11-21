"""
Calls the train_model.py script that is used to train the GPT model
using Rotary Positional Embeddings (RoPE).
"""

import train_model

if __name__ == "__main__":
    number_of_layers = [2, 4, 8, 12]  # train using varying numbers of transformer layers
    for layers in number_of_layers:
        train_model.train(layers)
