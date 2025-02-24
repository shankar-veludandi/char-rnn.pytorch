#!/usr/bin/env python
# https://github.com/spro/char-rnn.pytorch

import torch
import os
import argparse

from helpers import *
from model import *

def generate(decoder, prime_str='A', predict_len=100, temperature=0.8, cuda=False):
    hidden = decoder.init_hidden(1)
    prime_input = Variable(char_tensor(prime_str).unsqueeze(0))

    if cuda:
        hidden = hidden.cuda()
        prime_input = prime_input.cuda()

    predicted = prime_str

    # Use priming string to "build up" hidden state
    for p in range(len(prime_str) - 1):
        _, hidden = decoder(prime_input[:, p], hidden)
        
    inp = prime_input[:, -1]
    
    for p in range(predict_len):
        output, hidden = decoder(inp, hidden)
        
        # Sample from the network as a multinomial distribution
        output_dist = output.data.view(-1).div(temperature).exp()
        top_i = torch.multinomial(output_dist, 1)[0]

        # Add predicted character to string and use as next input
        predicted_char = all_characters[top_i]
        predicted += predicted_char
        inp = Variable(char_tensor(predicted_char).unsqueeze(0))
        if cuda:
            inp = inp.cuda()

    return predicted

if __name__ == '__main__':

    argparser = argparse.ArgumentParser()
    argparser.add_argument('filename', type=str)
    argparser.add_argument('-p', '--prime_str', type=str, default='A')
    argparser.add_argument('-l', '--predict_len', type=int, default=100)
    argparser.add_argument('-t', '--temperature', type=float, default=0.8)
    argparser.add_argument('--cuda', action='store_true')
    argparser.add_argument('--hidden_size', type=int, default=128)
    argparser.add_argument('--n_layers', type=int, default=2)
    args = argparser.parse_args()

    # 1) Create a CharRNN instance with the SAME parameters used during training
    #    e.g. if you trained with model='rnn', hidden_size=128, n_layers=2, etc.
    decoder = CharRNN(
        input_size=n_characters,
        hidden_size=args.hidden_size,
        output_size=n_characters,
        model=args.model_type,
        n_layers=args.n_layers
    )

    # 2) Load state dict from file (weights_only)
    state_dict = torch.load(args.filename, weights_only=True)
    decoder.load_state_dict(state_dict)

    # If using GPU
    if args.cuda:
        decoder.cuda()

    # 3) Generate text
    del args.filename
    print(generate(decoder, **vars(args)))
