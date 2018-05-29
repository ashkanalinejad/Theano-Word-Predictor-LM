from collections import OrderedDict
import cPickle as pkl
import sys
import time

import numpy
import theano
from theano import config
import theano.tensor as tensor
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from utils.data_iterator import TextIterator, WindowIterator

from LM import *

if __name__ == '__main__':

    test_data = ['/Users/alinejad/Desktop/SFU/SNMT-Prediction/Datasets/wmt16_en_de/newstest2015.tok.bpe.32000.en',
                 '/Users/alinejad/Desktop/SFU/SNMT-Prediction/Datasets/wmt16_en_de/predictnewstest.tok.bpe.32000.en']
    dictionaries = ['/Users/alinejad/Desktop/SFU/SNMT-Prediction/dl4mt-simul-trans/data/all_de-en.en.tok.bpe.pkl']

    model_options = OrderedDict()
    model_options['dim_proj'] = 1024
    model_options['patience'] = 10
    model_options['max_epochs'] = 5000
    model_options['dispFreq'] = 10
    model_options['decay_c'] = 0.
    model_options['lrate'] = 0.00002
    model_options['n_words'] = 20000
    model_options['encoder'] = 'lstm'
    model_options['validFreq'] = 1000
    model_options['saveFreq'] = 1000
    model_options['maxlen'] = 100
    model_options['batch_size'] = 64
    model_options['valid_batch_size'] = 64
    model_options['noise_std'] = 0.
    model_options['use_dropout'] = True
    model_options['reload_model'] = None
    model_options['test_size'] = -1
    model_options['ydim'] = 20000

    params = OrderedDict()
    params = init_params(model_options)
    load_params('lstm_model.npz', params)

    tparams = init_tparams(params)
    print "params created."
	
    (use_noise, x, mask,
     y, y_mask, f_pred_prob, f_pred, log_probs, f_u_shape) = build_model(tparams, model_options)
    print "model built."

    test = TextIterator(test_data[0], test_data[1], 
                         dictionaries[0],
                         n_words_source=20000, n_words_target=20000,
                         batch_size=model_options['batch_size'],
                         maxlen=model_options['maxlen'])
    print "test set is ready."

    print "Start Testing ..."
    use_noise.set_value(0.)
    train_err, correct, pp, num_windows, perplexity = pred_error(f_pred, f_pred_prob, prepare_data, model_options, test)
    print "final : ", train_err
    print "total : ", num_windows
    print "corrects   : ", correct[0]
    print "prediction : ", pp[0]
    print "perplexity : ", perplexity

