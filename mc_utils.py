import numpy as np

def process_input_index(data_raw,word2vec,input_mask_mode):
    inputs = []
    questions = []
    choices = []
    answers = []
    input_masks = []
    maxst = 0
    maxq = 0
    maxTc = 0
    for x in data_raw:
        inputs.append(np.array([w for s in x["C"] for w in s],dtype=np.int32))
        maxst  =max(maxst,len(inputs[-1]))
        questions.append(np.array(x["Q"],dtype=np.int32))
        maxq = max(maxq,len(questions[-1]))                
        answers.append(x["A"])
        choices.append([np.array(opt,dtype=np.int32) for opt in x["O"]])
        
        if input_mask_mode == 'word':
            input_masks.append(np.array(xrange(len(inp_vector)), dtype=np.int32)) 
        elif input_mask_mode == 'sentence':
            sentence_length = np.array([len(s) for s in x["C"]], dtype=np.int32)
            input_masks.append(np.cumsum(sentence_length,dtype=np.int32)-1) 
        else:
            raise Exception("invalid input_mask_mode")
        maxTc = max(maxTc,len(input_masks[-1]))
        
    print("max statement length is {}".format(maxst))
    print("max question length is {}".format(maxq))
    print("max Tc length is {}".format(maxTc))
    return inputs, questions, answers, choices, input_masks
    
def process_input_glove(data_raw, word2vec, input_mask_mode):
    inputs = []
    questions = []
    choices = []
    answers = []
    input_masks = []
    maxst = 0
    maxq = 0
    maxTc = 0
    for x in data_raw:
        inputs.append(np.vstack([word2vec[w] for s in x["C"] for w in s]).astype(np.float32)) #(seq_len, embed)
        maxst  =max(maxst,len(inputs[-1]))
        questions.append(np.vstack([word2vec[w] for w in x["Q"]]).astype(np.float32))
        maxq = max(maxq,len(questions[-1]))                
        answers.append(x["A"])
        choices.append([np.vstack([word2vec[w] for w in opt]).astype(np.float32) for opt in x["O"]])
        
        if input_mask_mode == 'word':
            input_masks.append(np.array(xrange(len(inp_vector)), dtype=np.int32)) 
        elif input_mask_mode == 'sentence':
            sentence_length = np.array([len(s) for s in x["C"]], dtype=np.int32)
            input_masks.append(np.cumsum(sentence_length,dtype=np.int32)-1) 
        else:
            raise Exception("invalid input_mask_mode")
        maxTc = max(maxTc,len(input_masks[-1]))
        
    print("max statement length is {}".format(maxst))
    print("max question length is {}".format(maxq))
    print("max Tc length is {}".format(maxTc))
    return inputs, questions, answers, choices, input_masks
