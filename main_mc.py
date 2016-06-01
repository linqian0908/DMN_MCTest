import sys
import numpy as np
import sklearn.metrics as metrics
import argparse
import time
import json

import utils
import nn_utils
import mctest_parse
import cPickle

print "==> parsing input arguments"
parser = argparse.ArgumentParser()

parser.add_argument('--network', type=str, default="dmn_qa_mc", help='network type: dmn_qa_mc or dmn_qa_batch')
parser.add_argument('--word_vector_size', type=int, default=50, help='embeding size (50, 100, 200, 300 only)')
parser.add_argument('--dim', type=int, default=40, help='number of hidden units in input module GRU')
parser.add_argument('--epochs', type=int, default=30, help='number of epochs')
parser.add_argument('--load_state', type=str, default="", help='state file path')
parser.add_argument('--answer_module', type=str, default="feedforward", help='answer module type: feedforward or recurrent')
parser.add_argument('--mode', type=str, default="train", help='mode: train or test. Test mode required load_state')
parser.add_argument('--input_mask_mode', type=str, default="sentence", help='input_mask_mode: word or sentence')
parser.add_argument('--memory_hops', type=int, default=5, help='memory GRU steps')
parser.add_argument('--batch_size', type=int, default=10, help='no commment')
parser.add_argument('--id', type=str, default="mc160", help='MCTest task ID')
parser.add_argument('--l2', type=float, default=0, help='L2 regularization')
parser.add_argument('--normalize_attention', type=bool, default=False, help='flag for enabling softmax on attention vector')
parser.add_argument('--log_every', type=int, default=1, help='print information every x iteration')
parser.add_argument('--save_every', type=int, default=1, help='save state every x epoch')
parser.add_argument('--prefix', type=str, default="", help='optional prefix of network name')
parser.add_argument('--no-shuffle', dest='shuffle', action='store_false')
parser.add_argument('--dropout', type=float, default=0.0, help='dropout rate (between 0 and 1)')
parser.add_argument('--batch_norm', type=bool, default=False, help='batch normalization')
parser.add_argument('--parse', type=bool, default=False, help='whether to load per-processed data and wor2vec')
parser.set_defaults(shuffle=True)
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

if not args.parse: # read preprossed data and embedding matrix
    print "loading data and embedding from pickle"
    train_raw, dev_raw, test_raw = mctest_parse.read_mc(args.id)
    word2vec = mctest_parse.read_embedding(args.id,args.word_vector_size)
else:
    print "parsing data and generating embedding"
    train_raw, dev_raw, test_raw, vocab = mctest_parse.build_mc(args.id)
    word2vec = mctest_parse.build_embedding(vocab,args.word_vector_size)

args_dict = dict(args._get_kwargs())
args_dict['train_raw'] = train_raw
args_dict['dev_raw'] = dev_raw
args_dict['test_raw'] = test_raw
args_dict['word2vec'] = word2vec

# init class
if args.network == 'dmn_qa_batch':
    import dmn_qa_batch
    dmn = dmn_batch.DMN_batch(**args_dict)

elif args.network == 'dmn_qa_mc':
    import dmn_qa_mc
    if (args.batch_size != 1):
        print "==> no minibatch training, argument batch_size is useless"
        args.batch_size = 1
    dmn = dmn_qa_mc.DMN_qa(**args_dict)

else: 
    raise Exception("No such network known: " + args.network)
    

if args.load_state != "":
    dmn.load_state(args.load_state)

def do_epoch(mode, epoch, skipped=0):
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
            
            for x in answers:
                y_true.append(x)
            
            for x in prediction.argmax(axis=1):
                y_pred.append(x)
            
            # TODO: save the state sometimes
            if (i % args.log_every == 0):
                cur_time = time.time()
                print ("  %sing: %d.%d / %d \t loss: %.3f \t avg_loss: %.3f \t skipped: %d \t %s \t time: %.2fs" % 
                    (mode, epoch, i * args.batch_size, batches_per_epoch * args.batch_size, 
                     current_loss, avg_loss / (i + 1), skipped, log, cur_time - prev_time))
                prev_time = cur_time
        
        if np.isnan(current_loss):
            print "==> current loss IS NaN. This should never happen :) " 
            exit()

    avg_loss /= batches_per_epoch
    print "\n  %s loss = %.5f" % (mode, avg_loss)
    print "confusion matrix:"
    print metrics.confusion_matrix(y_true, y_pred)
    
    accuracy = sum([1 if t == p else 0 for t, p in zip(y_true, y_pred)])
    print "accuracy: %.2f percent" % (accuracy * 100.0 / batches_per_epoch / args.batch_size)
    
    return avg_loss, skipped


if args.mode == 'train':
    print "==> training"   	
    skipped = 0
    for epoch in range(args.epochs):
        start_time = time.time()
        
        if args.shuffle:
            dmn.shuffle_train_set()
        
        _, skipped = do_epoch('train', epoch, skipped)
        
        epoch_loss, skipped = do_epoch('test', epoch, skipped)
        
        state_name = 'states/%s.epoch%d.test%.5f.state' % (network_name, epoch, epoch_loss)

        if (epoch % args.save_every == 0):    
            print "==> saving ... %s" % state_name
            dmn.save_params(state_name, epoch)
        
        print "epoch %d took %.3fs" % (epoch, float(time.time()) - start_time)

elif args.mode == 'test':
    file = open('last_tested_model.json', 'w+')
    data = dict(args._get_kwargs())
    data["id"] = network_name
    data["name"] = network_name
    data["description"] = ""
    data["vocab"] = dmn.vocab.keys()
    json.dump(data, file, indent=2)
    do_epoch('test', 0)

else:
    raise Exception("unknown mode")
