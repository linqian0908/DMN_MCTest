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

print "==> parsing input arguments"
parser = argparse.ArgumentParser()

parser.add_argument('--network', type=str, default="gru_gen", help='network type: gru_gen')
parser.add_argument('--word_vector_size', type=int, default=50, help='embeding size (50, 100, 200, 300 only)')
parser.add_argument('--dim', type=int, default=40, help='number of hidden units in input module GRU')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--load_state', type=str, default="", help='state file path')
parser.add_argument('--answer_module', type=str, default="recurrent", help='answer module type: feedforward or recurrent')
parser.add_argument('--mode', type=str, default="train", help='mode: train or test. Test mode required load_state')
parser.add_argument('--input_mask_mode', type=str, default="sentence", help='input_mask_mode: word or sentence')
parser.add_argument('--memory_hops', type=int, default=3, help='memory GRU steps')
parser.add_argument('--batch_size', type=int, default=10, help='no commment')
parser.add_argument('--id', type=str, default="mc160", help='MCTest task ID')
parser.add_argument('--l2', type=float, default=0.001, help='L2 regularization')
parser.add_argument('--normalize_attention', type=bool, default=False, help='flag for enabling softmax on attention vector')
parser.add_argument('--log_every', type=int, default=500, help='print information every x iteration')
parser.add_argument('--save_every', type=int, default=1, help='save state every x epoch')
parser.add_argument('--prefix', type=str, default="", help='optional prefix of network name')
parser.add_argument('--dropout', type=float, default=0, help='dropout rate (between 0 and 1)')
parser.add_argument('--batch_norm', type=bool, default=False, help='batch normalization')
parser.add_argument('--load_embed', type=bool, default=True, help='whether to load per-processed data and wor2vec')
parser.add_argument('--load_gate', type=str, default="", help='whether to load gate weights W1, W2, Wb, b1, b2 from babi supervised training. make sure dimension consistency between two models!')
parser.add_argument('--lr', type=float, default=1.0, help='Learning rate')
args = parser.parse_args()

print args

assert args.word_vector_size in [50, 100, 200, 300]

network_name = args.prefix + '%s.mh%d.n%d.bs%d%s%s%s.%s' % (
    args.network, 
    args.memory_hops, 
    args.dim, 
    args.batch_size, 
    ".na" if args.normalize_attention else "", 
    ".bn" if args.batch_norm else "", 
    (".d" + str(args.dropout)) if args.dropout>0 else "",
    args.id)

train_raw, dev_raw, test_raw, _, _, _, vocab = mctest_parse.build_mc(args.id)
if args.load_embed: # read preprossed data and embedding matrix
    word2vec = mctest_parse.read_embedding(args.id,args.word_vector_size)
else:
    word2vec = mctest_parse.build_embedding(vocab,args.word_vector_size)

ivocab = {}
for w in vocab:
    ivocab[vocab[w]] = w
    
args_dict = dict(args._get_kwargs())
args_dict['train_raw'] = train_raw
args_dict['dev_raw'] = dev_raw
args_dict['test_raw'] = test_raw
args_dict['word2vec'] = word2vec

# init class
if args.network == 'gru_gen':
    from mc_gru_gen_ans import DMN    
else: 
    raise Exception("No such network known: " + args.network)
    
if (args.batch_size != 1):
        print "==> no minibatch training, argument batch_size is useless"
        args.batch_size = 1
dmn = DMN(**args_dict)

if args.load_state != "":
    dmn.load_state(args.load_state)

if args.load_gate != "":
    dmn.load_gate(args.load_gate)
    
def sentence_equal(t,p):
    if not len(t)==len(p):
        print "===> predicted answer length ",len(p)," not equal true length ",len(t)
        exit()
    for x,y in zip(t,p):
        if not x==y:
            return False
    return True
            
def do_epoch(args, mode, epoch, skipped=0):
    # mode is 'train' or 'test'
    y_true = []
    y_pred = []
    avg_loss = 0.0
    prev_time = time.time()
    
    batches_per_epoch = dmn.get_batches_per_epoch(mode)
    
    for i in range(0, batches_per_epoch):
        step_data = dmn.step(i, mode)
        prediction = step_data["prediction"]
        answers = step_data["answers"]
        current_loss = step_data["current_loss"]
        current_skip = (step_data["skipped"] if "skipped" in step_data else 0)
        log = step_data["log"]
        
        skipped += current_skip
        
        if current_skip == 0:                
            avg_loss += current_loss
            y_true.append(answers)
            y_pred.append(prediction.argmax(axis=1))
            if random.random()<0.05:
                print 'True: '+' '.join(str(ivocab[x]) for x in y_true)
                print 'Predict: '+' '.join(str(vocab[x]) for x in y_pred)
                
            # TODO: save the state sometimes
            if (i % args.log_every == (args.log_every-1)):
                cur_time = time.time()
                print ("%sing: %d.%d/%d, loss: %.3f, avg_loss: %.3f, skipped: %d, %s, time: %.2fs" % 
                    (mode, epoch, i * args.batch_size, batches_per_epoch * args.batch_size, 
                     current_loss, avg_loss / (i + 1), skipped, log, cur_time - prev_time))
                prev_time = cur_time
        
        if np.isnan(current_loss):
            print "==> current loss IS NaN. This should never happen :) " 
            exit()

    avg_loss /= batches_per_epoch
    print "\n%s loss = %.5f" % (mode, avg_loss) 

    accuracy = sum([1 if sentence_equal(t,p) else 0 for t,p in zip(y_true,y_pred)])
    print "accuracy: %.2f percent" % (accuracy * 100.0 / batches_per_epoch / args.batch_size)
                
    return avg_loss, skipped

if args.mode == 'train':
    print "==> training"   	
    skipped = 0
    for epoch in range(args.epochs):
        start_time = time.time()        
        _, skipped = do_epoch(args, 'train', epoch, skipped)        
        epoch_loss, skipped = do_epoch(args, 'dev', epoch, skipped)        
        state_name = 'states/%s.epoch%d.test%.5f.state' % (network_name, epoch, epoch_loss)
        
        if (epoch % args.save_every == 0):    
            print "==> saving ... %s" % state_name
            dmn.save_params(state_name, epoch)        
        print "epoch %d took %.3fs" % (epoch, float(time.time()) - start_time)
    
    epoch_loss, skipped = do_epoch(args, 'test', epoch, skipped)
    
elif args.mode == 'test':
    file = open('last_tested_model.json', 'w+')
    data = dict(args._get_kwargs())
    data["id"] = network_name
    data["name"] = network_name
    data["description"] = ""
    data["vocab"] = vocab
    json.dump(data, file, indent=2)
    do_epoch(args,'test', 0)

else:
    raise Exception("unknown mode")
