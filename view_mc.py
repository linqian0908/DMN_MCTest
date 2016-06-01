import sys
import numpy as np
import sklearn.metrics as metrics
import argparse
import time
import json
import random

import utils
import nn_utils
import mctest_parse
import cPickle

from mctest_parse import print_words

print "==> parsing input arguments"
parser = argparse.ArgumentParser()

parser.add_argument('--network', type=str, default="gru_dot", help='network type: gru_pend, gru_dot, gru_pend_fix')
parser.add_argument('--word_vector_size', type=int, default=50, help='embeding size (50, 100, 200, 300 only)')
parser.add_argument('--dim', type=int, default=40, help='number of hidden units in input module GRU')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--load_state', type=str, default="", help='state file path')
parser.add_argument('--answer_module', type=str, default="feedforward", help='answer module type: feedforward or recurrent')
parser.add_argument('--mode', type=str, default="test", help='mode: train or test. Test mode required load_state')
parser.add_argument('--input_mask_mode', type=str, default="sentence", help='input_mask_mode: word or sentence')
parser.add_argument('--memory_hops', type=int, default=3, help='memory GRU steps')
parser.add_argument('--batch_size', type=int, default=10, help='no commment')
parser.add_argument('--id', type=str, default="mc160", help='MCTest task ID')
parser.add_argument('--l2', type=float, default=0.001, help='L2 regularization')
parser.add_argument('--normalize_attention', type=bool, default=False, help='flag for enabling softmax on attention vector')
parser.add_argument('--log_every', type=int, default=500, help='print information every x iteration')
parser.add_argument('--save_every', type=int, default=5, help='save state every x epoch')
parser.add_argument('--prefix', type=str, default="", help='optional prefix of network name')
parser.add_argument('--dropout', type=float, default=0, help='dropout rate (between 0 and 1)')
parser.add_argument('--batch_norm', type=bool, default=False, help='batch normalization')
parser.add_argument('--load_embed', type=bool, default=True, help='whether to load per-processed data and wor2vec')
parser.add_argument('--lr', type=float, default=1.0, help='Learning rate')
args = parser.parse_args()

print args

assert args.word_vector_size in [50, 100, 200, 300]

train_raw, dev_raw, test_raw, train_w, dev_w, test_w, vocab = mctest_parse.build_mc(args.id)
if args.load_embed: # read preprossed data and embedding matrix
    word2vec = mctest_parse.read_embedding(args.id,args.word_vector_size)
else:
    word2vec = mctest_parse.build_embedding(vocab,args.word_vector_size)

args_dict = dict(args._get_kwargs())
args_dict['train_raw'] = train_raw
args_dict['dev_raw'] = dev_raw
args_dict['test_raw'] = test_raw
args_dict['word2vec'] = word2vec

# init class
if args.network == 'gru_pend':
    from mc_gru_pend import DMN
elif args.network == 'gru_dot':
    from mc_gru_dot import DMN
elif args.network == 'gru_pend_fix':
    from mc_gru_pend_fix import DMN
elif args.network == 'gru_dot_fix':
    from mc_gru_dot_fix import DMN
else: 
    raise Exception("No such network known: " + args.network)
    
if (args.batch_size != 1):
        print "==> no minibatch training, argument batch_size is useless"
        args.batch_size = 1
dmn = DMN(**args_dict)

if args.load_state != "":
    dmn.load_state(args.load_state)

unseen_raw = dev_raw + test_raw
unseen_w = dev_w + test_w
input_str = ""
while not input_str=="exit":
    print "==> Training sample"
    n = random.randint(0,len(train_raw)-1)
    print_words(train_w[n])
    prob, attentions = dmn.predict([train_raw[n]])
    print '...Confidence: {}'.format(prob.max())
    print attentions
    
    print "==> Test/dev sample"
    n = random.randint(0,len(unseen_raw)-1)
    print_words(unseen_w[n])
    prob, attentions = dmn.predict([unseen_raw[n]])
    print '...Confidence: {}'.format(prob.max())
    print attentions
    
    input_str = raw_input("Press ENTER to continue. Type exit to stop: ")
