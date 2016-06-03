## adapted from parparthshah's mctest_dataset_parser_v2.py
import re
import sys, os
import cPickle
import random
import numpy as np
import codecs

def only_words(line):
    ps = re.sub(r'[^a-zA-Z0-9\']', r' ', line)
    ws = re.sub(r'(\W)', r' \1 ', ps) # Put spaces around punctuations
    #ws = re.sub(r" ' ", r"'", ws) # Remove spaces around '
    # ns = re.sub(r'(\d+)', r' <number> ', ws) # Put spaces around numbers
    hs = re.sub(r'-', r' ', ws) # Replace hyphens with space
    rs = re.sub(r' +', r' ', hs) # Reduce multiple spaces into 1
    rs = rs.lower().strip().split(' ')
    return rs

def clean_sentence(line):
    ps = re.sub(r'[^a-zA-Z0-9\.\?\!\']', ' ', line) # Split on punctuations and hex characters
    ws = re.sub(r'(\W)', r' \1 ', ps) # Put spaces around punctuations
    #ws = re.sub(r" ' ", r"'", ws) # Remove spaces around '
    # ns = re.sub(r'(\d+)', r' <number> ', ws) # Put spaces around numbers
    hs = re.sub(r'-', r' ', ws) # Replace hyphens with space
    rs = re.sub(r' +', r' ', hs) # Reduce multiple spaces into 1
    rs = rs.lower().strip()
    return rs

def get_sentences(line):
    ps = re.sub(r'[^a-zA-Z0-9\.\?\!\']', ' ', line) # Split on punctuations and hex characters
    s = re.sub(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', '\t', ps) # Split on sentences
    ws = re.sub(r'(\W)', r' \1 ', s) # Put spaces around punctuations
    #ws = re.sub(r" ' ", r"'", ws) # Remove spaces around '
    # ns = re.sub(r'(\d+)', r' <number> ', ws) # Put spaces around numbers
    hs = re.sub(r'-', r' ', ws) # Replace hyphens with space
    rs = re.sub(r' +', r' ', hs) # Reduce multiple spaces into 1
    rs = rs.lower().strip()
    return rs.split('\t')

def get_answer_index(a):
    answer_to_index = {
        'A': 0,
        'B': 1,
        'C': 2,
        'D': 3,
    }
    return answer_to_index[a]

def parse_mc(questions_file, answers_file, vocab={}):
    tasks_w = []
    tasks_id = []
    word_id = len(vocab)
     
    article_files = set()
    print("Parsing questions %s %s" % (questions_file, answers_file))
    q_file = open(questions_file, 'r')
    a_file = open(answers_file, 'r')

    questions_data = q_file.readlines()
    answers_data = a_file.readlines()
    assert(len(questions_data) == len(answers_data))

    for i in xrange(len(questions_data)):
        question_line = questions_data[i]
        answer_line = answers_data[i]

        question_pieces = question_line.strip().split('\t')
        assert(len(question_pieces) == 23)

        answer_pieces = answer_line.strip().split('\t')
        assert(len(answer_pieces) == 4)
        
        # parse story sentence by sentence. statement = list of sentence, sentence = list of word_index
        text = question_pieces[2]
        text = text.replace('\\newline', ' ')
        sentences = get_sentences(text) 
        statements_w = []                   
        statements_idx = []
        for s in sentences:
            tokens = s.strip().split()
            statements_w.append(tokens)
            indices = []
            # update word id
            for token in tokens:
                if token not in vocab:
                    vocab[token] = word_id
                    indices.append(word_id)
                    word_id += 1
                else:
                    indices.append(vocab[token])
            statements_idx.append(indices)
        
        # parsing four questions and their options for each story
        for j in range(4):
            q_index = (j * 5) + 3
            q_w = question_pieces[q_index]
            q_w = clean_sentence(q_w).split()
            assert(q_w[0] == 'multiple' or q_w[0] == 'one')
            del q_w[0]
            
            options_w = [
                only_words(question_pieces[q_index + 1]),
                only_words(question_pieces[q_index + 2]),
                only_words(question_pieces[q_index + 3]),
                only_words(question_pieces[q_index + 4]),
            ]
            correct = get_answer_index(answer_pieces[j])

            # parse question
            q_idx = []
            for token in q_w:
                if token not in vocab:
                    vocab[token] = word_id
                    q_idx.append(word_id)
                    word_id += 1
                else:
                    q_idx.append(vocab[token])
                    
            # parse options                
            options_idx = []
            for o in options_w:
                o_idx = []
                for w in o:
                    if w not in vocab:
                        vocab[w] = word_id
                        o_idx.append(word_id)
                        word_id += 1
                    else:
                        o_idx.append(vocab[w])
                options_idx.append(o_idx)
            
            tasks_id.append({"C":statements_idx,"Q":q_idx,"O":options_idx,"A":correct})
            tasks_w.append({"C":statements_w,"Q":q_w,"O":options_w,"A":correct})

    print "There are %d tasks" % len(tasks_id)
    print "There are %d words" % len(vocab)
    return {'idata':tasks_id, 'wdata':tasks_w, 'vocab':vocab}

def print_words(task):
    print "...Story..."
    story = task["C"];
    for i in xrange(len(story)):
        print ' '.join(story[i])
    
    print "...Question..."
    print ' '.join(task["Q"])
    
    print "...Options..."
    options = task["O"]
    for i in xrange(len(options)):
        print ' '.join(options[i])
    
    print "...Correct answer..."
    print task["A"]

def print_indices(task):
    print "Story..."
    story = task["C"];
    for i in xrange(len(story)):
        print ' '.join(str(x) for x in story[i])
    
    print "Question..."
    print ' '.join(str(x) for x in task["Q"])
    
    print "Options..."
    options = task["O"]
    for i in xrange(len(options)):
        print ' '.join(str(x) for x in options[i])
    
    print "Correct answer..."
    print task["A"]
    
def build_mc(id): 
    data_dir = 'data/MCTest'
    train_file = id+'.train.tsv'
    print "Train file:", train_file
    train_answers = train_file.replace('tsv', 'ans')
    dev_file = train_file.replace('train','dev')
    dev_answers = dev_file.replace('tsv','ans')
    test_file = train_file.replace('train', 'test')
    test_answers = test_file.replace('tsv', 'ans')
    
    train_obj = parse_mc(os.path.join(data_dir, train_file), os.path.join(data_dir, train_answers))
    dev_obj = parse_mc(os.path.join(data_dir, dev_file),os.path.join(data_dir, dev_answers),vocab = train_obj['vocab'])  
    test_obj = parse_mc(os.path.join(data_dir, test_file), os.path.join(data_dir, test_answers), vocab = dev_obj['vocab'])
    
    return train_obj['idata'], dev_obj['idata'], test_obj['idata'], train_obj['wdata'], dev_obj['wdata'], test_obj['wdata'], test_obj['vocab']
    
#Given a word-to-id dictionary, return embedding initialized to Glove. word not found in Glove will be initialized to random
def build_embedding(word_to_id, vocab_dim):
    vector_file = "data/glove/glove.6B." + str(vocab_dim) + "d.txt"
    n_words = len(word_to_id)
    embedding_weights = np.random.uniform(0.0,1.0,(n_words, vocab_dim)).astype(np.float32)
    found = np.zeros(n_words)
    with codecs.open(vector_file, 'r', 'utf-8') as f:
        for c, r in enumerate(f):
            sr = r.split()
            if sr[0] in word_to_id:
                embedding_weights[word_to_id[sr[0]]] = np.array([float(i) for i in sr[1:]])
                found[word_to_id[sr[0]]] = 1
    for w in word_to_id:
        if not found[word_to_id[w]]:
            print w
    
    print 'Loading glove... total words: {}'.format(n_words)
    print 'Words loaded from glove: {}'.format(np.sum(found))
    return embedding_weights

def read_embedding(id,dim):
    data_dir = 'data/MCTest'
    print("Loading pickled embedding")
    f = file(os.path.join(data_dir, id+'.'+str(dim)+'d.pickle'), 'rb')
    embed = cPickle.load(f)
    return np.array(embed,dtype=np.float32)

if __name__ == "__main__":    
    data_dir = "data/MCTest"
    dataset = sys.argv[1]
    vocab_dim = int(sys.argv[2])

    train_file = dataset + '.train.tsv'
    print "Train file:", train_file
    train_answers = train_file.replace('tsv', 'ans')
    dev_file = train_file.replace('train','dev')
    dev_answers = dev_file.replace('tsv','ans')
    test_file = train_file.replace('train', 'test')
    test_answers = test_file.replace('tsv', 'ans')

    train_obj = parse_mc(os.path.join(data_dir, train_file), os.path.join(data_dir, train_answers))
    dev_obj = parse_mc(os.path.join(data_dir, dev_file),os.path.join(data_dir, dev_answers),vocab = train_obj['vocab'])  
    test_obj = parse_mc(os.path.join(data_dir, test_file), os.path.join(data_dir, test_answers), vocab = dev_obj['vocab'])
    
    # examine data
    n = random.randint(0,len(train_obj['idata'])-1)
    print("Train data {}".format(n))
    print_words(train_obj['wdata'][n])
    
    # Pickle!!!!
    print "Process embedding"
    embed = build_embedding(test_obj['vocab'], vocab_dim)
    
    embed_pickle = dataset+'.'+str(vocab_dim)+'d.pickle'
    print("Pickling embedding... " + embed_pickle)
    f = file(os.path.join(data_dir, embed_pickle), 'wb')
    cPickle.dump(embed, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
        
    print "Reading embedding"
    word2vec = read_embedding(dataset,vocab_dim)
    print(word2vec.shape)
    print(word2vec[0])
