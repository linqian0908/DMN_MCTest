# Dynamic memory networks in Theano
The aim of this repository is to implement Dynamic memory networks 
as described in the [paper by Kumar et al.](http://arxiv.org/abs/1506.07285)
and to extend it to mctest. DMN implementation for babi is based on https://github.com/YerevaNN/Dynamic-memory-networks-in-Theano. Our contribution include: adding supervised learning for babi, implementing recurrent answer module, dmn and parser for MCTest with options module, and visualization tools for episdoic memory.

## Repository contents for babi
| file | description |
| --- | --- |
| `main.py` | the main entry point to train and test available network architectures on bAbI-like tasks. modified from YerevaNN |
| `dmn_qa_draft.py` | draft version of a DMN designed for answering multiple choice questions | 
| `dmn_spv.py` | implement gate supervision as suggested in the Kumar paper, and enforce the attentions shift (including a special end_reading gate) from episdoe to next, except that we fixed the memory hops instead of conditional ending on "end_reading". uses the square of the Euclidean distance instead of `abs` in the attention module. neet to add l2 regularization for stable training (--l2 0.001). added gradient check to skip bad ones (NaN), but slows down the network. with regularization set to non-zero, might be able to get rid of the gradient check part for speedup. also include a main function for visualizing weights, which takes path to state to load |
| `utils.py` | tools for working with bAbI tasks and GloVe vectors. modified from YerevaNN |
| `nn_utils.py` | helper functions on top of Theano and Lasagne. copied from YerevaNN |

## Repository contents for MCTest

no mini-batch implementation. the _fix substript in network means the word embedding is loaded from glove with unknown set to 0 and is not retrained during training. those without _fix uses embedding matrix initialized from glove but allowed to retrain during training. 

| file | description |
| --- | --- |
| `main_mc.py` | the main entry point to train and test available network architectures on MCTest tasks |
| `mc_gru_dot_fix.py` | choices are dot-product with fianl memory state m in answer module. basic idea is a cos-similarity |
| `mc_gru_dot.py`  | see above |
| `mc_gru_pend_fix.py` | choice used in attention. this closely resemble dmn_qa_draft in architect. | 
| `mc_gru_pend.py` | see above | 
| `mctest_parse.py` | tools for parsing from raw mctest data. can be used to preprocessing word2vec embedding and save to memory |

## Repository contents (our addition for visualization)

no mini-batch implementation. the _fix substript in network means the word embedding is loaded from glove with unknown set to 0 and is not retrained during training. those without _fix uses embedding matrix initialized from glove but allowed to retrain during training. 

| file | description |
| --- | --- |
| `view_mc.py` | visualize attention gate at each episode. |
| `view_babi.py` | visualize attention gate at each episode. |

## Repository contents (files copied from YeveraNN/DMN in Theano, kept around for reference by not used)

| file | description |
| --- | --- |
| `dmn_basic.py` | our baseline implementation. It is as close to the original as we could understand the paper, except the number of steps in the main memory GRU is fixed. Attention module uses `T.abs_` function as a distance between two vectors which causes gradients to become `NaN` randomly.  The results reported in [this blog post](http://yerevann.github.io/2016/02/05/implementing-dynamic-memory-networks/) are based on this network |
| `dmn_smooth.py` | uses the square of the Euclidean distance instead of `abs` in the attention module. Training is very stable. Performance on bAbI is slightly better |
| `dmn_batch.py` | `dmn_smooth` with minibatch training support. The batch size cannot be set to `1` because of the [Theano bug](https://github.com/Theano/Theano/issues/1772) | 
| `dmn_qa_draft.py` | draft version of a DMN designed for answering multiple choice questions | 
| `fetch_babi_data.sh` | shell script to fetch bAbI tasks (adapted from [MemN2N](https://github.com/npow/MemN2N)) |
| `fetch_glove_data.sh` | shell script to fetch GloVe vectors (by [5vision](https://github.com/5vision/kaggle_allen)) |
| `server/` | contains Flask-based restful api server |

## DMN on babi Usage

This implementation is based on Theano and Lasagne. One way to install them is:

    pip install -r https://raw.githubusercontent.com/Lasagne/Lasagne/master/requirements.txt
    pip install https://github.com/Lasagne/Lasagne/archive/master.zip

The following bash scripts will download bAbI tasks and GloVe vectors.

    ./fetch_babi_data.sh
    ./fetch_glove_data.sh

Use `main.py` to train a network:

    python main.py --network dmn_basic --babi_id 1
    python main.py --network dmn_spv --babi_id 3 --memory_hops 5 --l2 0.001 --log_every 1000
    python main.py --network dmn_spv --babi_id 3 --memory_hops 5 --l2 0.001 --log_every 200 --load_state states/dmn_spv.mh5.n40.bs10.babi3.epoch5.test2.28071.state

The states of the network will be saved in `states/` folder. 
There is one pretrained state on the 1st bAbI task. It should give 100% accuracy on the test set:

    python main.py --network dmn_basic --mode test --babi_id 1 --load_state states/dmn_basic.mh5.n40.babi1.epoch4.test0.00033.state

To view weights of pre-trained state
    
    python dmn_spv.py states/dmn_spv.mh3.n40.bs10.babi2.epoch12.test1.16323.state

## MCTest data preprocessing:
download MCTest data into data/MCTest.
in root folder, run
    
    python mctest_parse.py [mc160|mc500] [50|100|200|300]
    
this generates the embedding matrix for mc160 and mc500, from glove.[50] ect.

the parser also have a function called by main_mc to generate a parsed dataset (converted to index). The returned data structure is a list of {"C":[[w]],"Q":[w],"A":[w],"O":[[w]]} for each question task

## MCTest run
run the main mctest network training
    
    python main_mc.py --network gru_dot_fix --id mc160

see the main_mc function for a list of options

## Visualize episode memory
view_babi and view_mc can visualize attention gate over episode. Need to load from pretrain model. Currently only dmn_smooth, dmn_batch, mc_gru_dot_fix network supports viewing.

    python view_babi.py --network dmn_smooth --babi_id 2 --load_state states/dmn_smooth.mh3.n40.bs10.babi2.epoch29.test6.43988.state
    
    sudo python view_babi.py --network dmn_spv --babi_id 2 --memory_hops 3 --load_state states/dmn_spv.mh3.n40.bs10.babi2.epoch12.test1.16323.state
    
    sudo python view_babi.py --network dmn_spv --babi_id 3 --memory_hops 5 --load_state states/dmn_spv.mh5.n40.bs10.babi3.epoch5.test2.28071.state
    
    python view_mc.py --network gru_dot_fix --id mc160 --load_state states/gru_dot_fix.mh3.n40.bs10.d0.3.mc160.epoch25.test5.22941.state
    
    sudo python view_babi.py --network dmn_spv --memory_hops 5 --babi_id 3 --load_state states/dmn_spv.mh5.n40.bs10.babi3.epoch29.test7.27444.state

## Roadmap

Supervised training on babi
Attention gate trained on babi and transfer to mctest
MCtest with recurrent answer module and direct answer generation (ignore choices at test time)

