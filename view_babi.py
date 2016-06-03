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
import pylab as plt

print "==> parsing input arguments"
parser = argparse.ArgumentParser()

parser.add_argument('--network', type=str, default="dmn_smooth", help='network type: dmn_batch, dmn_basic, dmn_smooth')
parser.add_argument('--word_vector_size', type=int, default=50, help='embeding size (50, 100, 200, 300 only)')
parser.add_argument('--dim', type=int, default=40, help='number of hidden units in input module GRU')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--load_state', type=str, default="", help='state file path')
parser.add_argument('--answer_module', type=str, default="feedforward", help='answer module type: feedforward or recurrent')
parser.add_argument('--mode', type=str, default="test", help='mode: train or test. Test mode required load_state')
parser.add_argument('--input_mask_mode', type=str, default="sentence", help='input_mask_mode: word or sentence')
parser.add_argument('--memory_hops', type=int, default=3, help='memory GRU steps')
parser.add_argument('--batch_size', type=int, default=10, help='no commment')
parser.add_argument('--babi_id', type=str, default="1", help='Babi ID 1-20')
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

babi_train_raw, babi_test_raw = utils.get_babi_raw(args.babi_id, args.babi_id)

word2vec = utils.load_glove(args.word_vector_size)

args_dict = dict(args._get_kwargs())
args_dict['babi_train_raw'] = babi_train_raw
args_dict['babi_test_raw'] = babi_test_raw
args_dict['word2vec'] = word2vec

# init class
dmn, batch_size = utils.get_dmn(args.network,args.batch_size,args_dict)
args.batch_size = batch_size
    
if (args.batch_size != 1):
        print "==> no minibatch training, argument batch_size is useless"
        args.batch_size = 1

if args.load_state != "":
    dmn.load_state(args.load_state)
    
input_str = ""
while not input_str=="exit":
    print "==> Training sample"
    n = random.randint(0,len(babi_train_raw)-1)
    ans, prob, attentions = dmn.predict([babi_train_raw[n]])
    if ans == babi_train_raw[n]["A"]:
        print "Correct!"
    else:
        print "Wrong :("
    print '...Prediction: {}'.format(ans)
    print '...Confidence: {}'.format(prob.max())
    print attentions
    plt.figure(0)
    plt.imshow(attentions)
    plt.show()
    
    print "==> Test/dev sample"
    n = random.randint(0,len(babi_test_raw)-1)
    ans, prob, attentions = dmn.predict([babi_test_raw[n]])    
    if ans == babi_test_raw[n]["A"]:
        print "Correct!"
    else:
        print "Wrong :("
    print '...Prediction: {}'.format(ans)
    print '...Confidence: {}'.format(prob.max())
    print attentions
    plt.figure(1)
    plt.imshow(attentions)
    plt.show()
    
    input_str = raw_input("Press ENTER to continue. Type exit to stop: ")
