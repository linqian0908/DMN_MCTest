import random
import numpy as np

import theano
import theano.tensor as T
from theano.compile.nanguardmode import NanGuardMode

import lasagne
from lasagne import layers
from lasagne import nonlinearities
import cPickle as pickle

import utils
import nn_utils
import mc_utils
import mctest_parse

floatX = theano.config.floatX

class DMN:
    
    def __init__(self, train_raw, dev_raw, test_raw, word2vec, word_vector_size, answer_module, 
                dim, mode, input_mask_mode, memory_hops, l2, normalize_attention, dropout, **kwargs):
        print "generate sentence answer for mctest"
        print "==> not used params in DMN class:", kwargs.keys()
        self.word2vec = word2vec      
        self.word_vector_size = word_vector_size
        # add eng_of_sentence tag for answer generation
        self.end_tag = len(word2vec)
        self.vocab_size = self.end_tag+1
        
        self.dim = dim # hidden state size
        self.mode = mode
        self.input_mask_mode = input_mask_mode
        self.memory_hops = memory_hops
        self.answer_module = answer_module
        self.l2 = l2
        self.normalize_attention = normalize_attention
        self.dropout = dropout
        
        self.train_input, self.train_q, self.train_answer, self.train_input_mask, self.train_max_n = self._process_input(train_raw)
        self.dev_input, self.dev_q, self.dev_answer, self.dev_input_mask, self.dev_max_n = self._process_input(dev_raw)
        self.test_input, self.test_q, self.test_answer, self.test_input_mask, self.test_max_n = self._process_input(test_raw)
        
        self.input_var = T.matrix('input_var')
        self.q_var = T.matrix('question_var')
        self.answer_var = T.ivector('answer_var')
        self.input_mask_var = T.ivector('input_mask_var')
        self.max_n = T.iscalar('max_n')
        
        self.attentions = []
            
        print "==> building input module"
        self.W_inp_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.W_inp_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inp_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.W_inp_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_inp_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.word_vector_size))
        self.W_inp_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_inp_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        inp_c_history, _ = theano.scan(fn=self.input_gru_step, 
                    sequences=self.input_var,
                    outputs_info=T.zeros_like(self.b_inp_hid))
        
        self.inp_c = inp_c_history.take(self.input_mask_var, axis=0)
        
        self.q_q, _ = theano.scan(fn=self.input_gru_step, 
                    sequences=self.q_var,
                    outputs_info=T.zeros_like(self.b_inp_hid))

        self.q_q = self.q_q[-1]
        
        
        print "==> creating parameters for memory module"
        self.W_mem_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_mem_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_mem_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_mem_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.b_mem_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
        self.W_b = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
        self.W_1 = nn_utils.normal_param(std=0.1, shape=(self.dim, 7 * self.dim + 2))
        self.W_2 = nn_utils.normal_param(std=0.1, shape=(1, self.dim))
        self.b_1 = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        self.b_2 = nn_utils.constant_param(value=0.0, shape=(1,))


        print "==> building episodic memory module (fixed number of steps: %d)" % self.memory_hops
        memory = [self.q_q.copy()]
        for iter in range(1, self.memory_hops + 1):
            current_episode = self.new_episode(memory[iter - 1])
            memory.append(self.GRU_update(memory[iter - 1], current_episode,
                                          self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                                          self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                                          self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid))
        
        last_mem_raw = memory[-1].dimshuffle(('x', 0))
        
        net = layers.InputLayer(shape=(1, self.dim), input_var=last_mem_raw)
        if self.dropout > 0 and self.mode == 'train':
            net = layers.DropoutLayer(net, p=self.dropout)
        last_mem = layers.get_output(net)[0]
        self.attentions = T.stack(self.attentions)
        
        print "==> building answer module"
        self.W_a = nn_utils.normal_param(std=0.1, shape=(self.vocab_size, self.dim))
        
        if self.answer_module == 'feedforward':
            self.prediction = nn_utils.softmax(T.dot(self.W_a, last_mem))
            self.prediction = self.prediction.dimshuffle('x',0)
            
        elif self.answer_module == 'recurrent':
            self.W_ans_res_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_res_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_res = nn_utils.constant_param(value=0.0, shape=(self.dim,))
            
            self.W_ans_upd_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_upd_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_upd = nn_utils.constant_param(value=0.0, shape=(self.dim,))
            
            self.W_ans_hid_in = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim + self.vocab_size))
            self.W_ans_hid_hid = nn_utils.normal_param(std=0.1, shape=(self.dim, self.dim))
            self.b_ans_hid = nn_utils.constant_param(value=0.0, shape=(self.dim,))
        
            def answer_step(prev_a, prev_y):
                a = self.GRU_update(prev_a, T.concatenate([prev_y, self.q_q]),
                                  self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
                                  self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                                  self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid)
                                  
                y = nn_utils.softmax(T.dot(self.W_a, a))
                return [a, y]#, theano.scan_module.until(n>=max_n)) # or argmax==self.end_tag)
            
            dummy = theano.shared(np.zeros((self.vocab_size, ), dtype=floatX))
            results, updates = theano.scan(fn=answer_step,
                               outputs_info=[last_mem, T.zeros_like(dummy)],
                               n_steps = self.max_n)
            self.prediction = results[1]       
        else:
            raise Exception("invalid answer_module")
        
        
        print "==> collecting all parameters"
        self.params = [self.W_inp_res_in, self.W_inp_res_hid, self.b_inp_res, 
                  self.W_inp_upd_in, self.W_inp_upd_hid, self.b_inp_upd,
                  self.W_inp_hid_in, self.W_inp_hid_hid, self.b_inp_hid,
                  self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                  self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                  self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid, self.W_b,
                  self.W_1, self.W_2, self.b_1, self.b_2, self.W_a]
        
        if self.answer_module == 'recurrent':#feedforward':
            self.params = self.params + [self.W_ans_res_in, self.W_ans_res_hid, self.b_ans_res, 
                              self.W_ans_upd_in, self.W_ans_upd_hid, self.b_ans_upd,
                              self.W_ans_hid_in, self.W_ans_hid_hid, self.b_ans_hid]
        
        
        print "==> building loss layer and computing updates"
        self.loss_ce = T.nnet.categorical_crossentropy(self.prediction,self.answer_var).sum()

        if self.l2 > 0:
            self.loss_l2 = self.l2 * nn_utils.l2_reg(self.params)
        else:
            self.loss_l2 = 0
        
        self.loss = self.loss_ce + self.loss_l2
        
        updates = lasagne.updates.adadelta(self.loss, self.params)
        #updates = lasagne.updates.momentum(self.loss, self.params, learning_rate=0.0003)
        
        if self.mode == 'train':
            print "==> compiling train_fn"
            self.train_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var, self.input_mask_var, self.max_n], 
                                            allow_input_downcast = True,
                                            outputs=[self.prediction, self.loss],
                                            updates=updates)
        
        print "==> compiling test_fn"
        self.test_fn = theano.function(inputs=[self.input_var, self.q_var, self.answer_var, self.input_mask_var, self.max_n],
                                       allow_input_downcast = True,
                                       outputs=[self.prediction, self.loss, self.attentions])
        
    
    def GRU_update(self, h, x, W_res_in, W_res_hid, b_res,
                         W_upd_in, W_upd_hid, b_upd,
                         W_hid_in, W_hid_hid, b_hid):
        """ mapping of our variables to symbols in DMN paper: 
        W_res_in = W^r
        W_res_hid = U^r
        b_res = b^r
        W_upd_in = W^z
        W_upd_hid = U^z
        b_upd = b^z
        W_hid_in = W
        W_hid_hid = U
        b_hid = b^h
        """
        z = T.nnet.sigmoid(T.dot(W_upd_in, x) + T.dot(W_upd_hid, h) + b_upd)
        r = T.nnet.sigmoid(T.dot(W_res_in, x) + T.dot(W_res_hid, h) + b_res)
        _h = T.tanh(T.dot(W_hid_in, x) + r * T.dot(W_hid_hid, h) + b_hid)
        return z * h + (1 - z) * _h   
    
    def input_gru_step(self, x, prev_h):
        return self.GRU_update(prev_h, x, self.W_inp_res_in, self.W_inp_res_hid, self.b_inp_res, 
                                     self.W_inp_upd_in, self.W_inp_upd_hid, self.b_inp_upd,
                                     self.W_inp_hid_in, self.W_inp_hid_hid, self.b_inp_hid)
    
    def new_attention_step(self, ct, prev_g, mem, q_q):
        cWq = T.stack([T.dot(T.dot(ct, self.W_b), q_q)])
        cWm = T.stack([T.dot(T.dot(ct, self.W_b), mem)])
        z = T.concatenate([ct, mem, q_q, ct * q_q, ct * mem, (ct - q_q) ** 2, (ct - mem) ** 2, cWq, cWm])
        
        l_1 = T.dot(self.W_1, z) + self.b_1
        l_1 = T.tanh(l_1)
        l_2 = T.dot(self.W_2, l_1) + self.b_2
        G = T.nnet.sigmoid(l_2)[0]
        return G
        
        
    def new_episode_step(self, ct, g, prev_h):
        gru = self.GRU_update(prev_h, ct,
                             self.W_mem_res_in, self.W_mem_res_hid, self.b_mem_res, 
                             self.W_mem_upd_in, self.W_mem_upd_hid, self.b_mem_upd,
                             self.W_mem_hid_in, self.W_mem_hid_hid, self.b_mem_hid)
        
        h = g * gru + (1 - g) * prev_h
        return h
       
    
    def new_episode(self, mem):
        g, g_updates = theano.scan(fn=self.new_attention_step,
            sequences=self.inp_c,
            non_sequences=[mem, self.q_q],
            outputs_info=T.zeros_like(self.inp_c[0][0])) 
        
        if (self.normalize_attention):
            g = nn_utils.softmax(g)

        self.attentions.append(g)
        
        e, e_updates = theano.scan(fn=self.new_episode_step,
            sequences=[self.inp_c, g],
            outputs_info=T.zeros_like(self.inp_c[0]))
        
        return e[-1]

    
    def save_params(self, file_name, epoch, **kwargs):
        with open(file_name, 'w') as save_file:
            pickle.dump(
                obj = {
                    'params' : [x.get_value() for x in self.params],
                    'epoch' : epoch, 
                    'gradient_value': (kwargs['gradient_value'] if 'gradient_value' in kwargs else 0)
                },
                file = save_file,
                protocol = -1
            )
    
    
    def load_state(self, file_name):
        print "==> loading state %s" % file_name
        with open(file_name, 'r') as load_file:
            dict = pickle.load(load_file)
            loaded_params = dict['params']
            for (x, y) in zip(self.params, loaded_params):
                x.set_value(y)

    
    def _process_input(self, data_raw):
        inputs = []
        questions = []
        answers = []
        input_masks = []
        maxst = 0
        maxq = 0
        maxTc = 0
        max_n = []
        for x in data_raw:
            inputs.append(np.vstack([self.word2vec[w] for s in x["C"] for w in s]).astype(np.float32)) #(seq_len, embed)
            maxst = max(maxst,len(inputs[-1]))
            questions.append(np.vstack([self.word2vec[w] for w in x["Q"]]).astype(np.float32)) #(q_len, embed)
            maxq = max(maxq,len(questions[-1]))
            option = [w for w in x["O"][x["A"]]]
            option.append(self.end_tag)
            max_n.append(len(option))
            answers.append(np.array(option,dtype=np.int32)) # (ans_len)
            
            if self.input_mask_mode == 'word':
                input_masks.append(np.array(xrange(len(inp_vector)), dtype=np.int32)) 
            elif self.input_mask_mode == 'sentence':
                sentence_length = np.array([len(s) for s in x["C"]], dtype=np.int32)
                input_masks.append(np.cumsum(sentence_length,dtype=np.int32)-1)  # (num_sentence)
            else:
                raise Exception("invalid input_mask_mode")
            maxTc = max(maxTc,len(input_masks[-1]))
           
        print("max statement length is {}".format(maxst))
        print("max question length is {}".format(maxq))
        print("max Tc length is {}".format(maxTc))
        print("max answer length is {}".format(max(max_n)))
        return inputs, questions, answers, input_masks, max_n

    def get_batches_per_epoch(self, mode):
        if (mode == 'train'):
            return len(self.train_input)
        elif (mode == 'dev'):
            return len(self.dev_input)
        elif (mode == 'test'):
            return len(self.test_input)
        else:
            raise Exception("unknown mode")
               
    def shuffle_train_set(self):
        print "==> Shuffling the train set"
        combined = zip(self.train_input, self.train_q, self.train_answer, self.train_input_mask)
        random.shuffle(combined)
        self.train_input, self.train_q, self.train_answer, self.train_input_mask = zip(*combined)
    
    
    def step(self, batch_index, mode):
        if mode == "train" and self.mode == "test":
            raise Exception("Cannot train during test mode")
        
        if mode == "train":
            theano_fn = self.train_fn 
            inputs = self.train_input
            qs = self.train_q
            answers = self.train_answer
            input_masks = self.train_input_mask
            max_ns = self.train_max_n
        elif mode == "test":    
            theano_fn = self.test_fn 
            inputs = self.test_input
            qs = self.test_q
            answers = self.test_answer
            input_masks = self.test_input_mask
            max_ns = self.test_max_n
        elif mode == "dev":    
            theano_fn = self.test_fn 
            inputs = self.dev_input
            qs = self.dev_q
            answers = self.dev_answer
            input_masks = self.dev_input_mask
            max_ns = self.dev_max_n
        else:
            raise Exception("Invalid mode")
            
        inp = inputs[batch_index]
        q = qs[batch_index]
        ans = answers[batch_index]
        input_mask = input_masks[batch_index]
        max_n = max_ns[batch_index]

        ret = theano_fn(inp, q, ans, input_mask, max_n)
        param_norm = np.max([utils.get_norm(x.get_value()) for x in self.params])
        
        return {"prediction": np.array([ret[0]]),
                "answers": np.array([ans]),
                "current_loss": ret[1],
                "skipped": 0,
                "log": "pn: %.3f" % param_norm,
                }
                
                
    def predict(self, data):
        # data is an array of objects like {"Q": "question", "C": "sentence ."}
        #data[0]["A"] = "."
        print "==> predicting:", data
        inputs, questions, answers, input_masks ,max_n = self._process_input(data)
        probabilities, loss, attentions = self.test_fn(inputs[0], questions[0], answers[0], input_masks[0], max_n[0])
        ans = self.ivocab[probabilities.argmax()]
        return ans, probabilities, attentions

