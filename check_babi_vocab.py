import utils
import numpy as np

def _process_input(data_raw, word2vec, vocab, ivocab, word_vector_size):
    sent_len = []
    gate_len = []
    for x in data_raw:
        inp = x["C"].lower().split(' ') 
        inp = [w for w in inp if len(w) > 0]
        q = x["Q"].lower().split(' ')
        q = [w for w in q if len(w) > 0]
        
        inp_vector = [utils.process_word(word = w.lower(), 
                                    word2vec = word2vec, 
                                    vocab = vocab, 
                                    ivocab = ivocab, 
                                    word_vector_size = word_vector_size, 
                                    to_return = "word2vec") for w in inp]
        sent_len.append(len(inp_vector))                            
        q_vector = [utils.process_word(word = w.lower(), 
                                    word2vec = word2vec, 
                                    vocab = vocab, 
                                    ivocab = ivocab, 
                                    word_vector_size = word_vector_size, 
                                    to_return = "word2vec") for w in q]        
        utils.process_word(word = x["A"].lower(), # TODO: add .lower() here 
                                    word2vec = word2vec, 
                                    vocab = vocab, 
                                    ivocab = ivocab, 
                                    word_vector_size = word_vector_size, 
                                    to_return = "index")
        gate_len.append(len(x["S"]))                       
    return sent_len, gate_len
    
word_vector_size=50
word2vec = utils.load_glove(word_vector_size)

for babi_id in range(1,21):
    babi_id = str(babi_id)
    print "processing babi."+babi_id
    vocab = {}
    ivocab = {}
    babi_train_raw, babi_test_raw = utils.get_babi_raw(babi_id, babi_id)
    train_len, train_gate = _process_input(babi_train_raw,word2vec,vocab,ivocab,word_vector_size)
    test_len, test_gate = _process_input(babi_test_raw,word2vec,vocab,ivocab,word_vector_size)
    print len(vocab)
    print max(train_gate), min(train_gate), max(test_gate), min(test_gate)
    print max(train_len),min(train_len),max(test_len),min(test_len)
