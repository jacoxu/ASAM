# -*- coding: utf-8 -*-
"""
    Initialize the parameters of model
"""
import config
import prepare_data
import numpy as np
__author__ = '[jacoxu](https://github.com/jacoxu)'


class ModelInit(object):
    def __init__(self, _log_file):
        # pre-process data
        print("Start to prepare data set")
        # map spk to idx
        spk_to_idx, idx_to_spk = prepare_data.get_idx(config.TRAIN_LIST)

        self.train_gen = prepare_data.get_feature(config.TRAIN_LIST, spk_to_idx=spk_to_idx, min_mix=config.MIN_MIX,
                                                  max_mix=config.MAX_MIX, batch_size=config.BATCH_SIZE)

        print("Finished data set preparation")
        self.inp_fea_len, self.inp_fea_dim, self.inp_spec_dim, self.inp_spk_len, self.out_spec_dim = \
            prepare_data.get_dims(self.train_gen)

        self.spk_size = len(spk_to_idx) + 1
        self.spk_to_idx = spk_to_idx
        self.idx_to_spk = idx_to_spk

    def init_spke_memory(self):
        # init Life-long Memory (spk_size, embed)
        return np.zeros(self.spk_size*config.EMBEDDING_SIZE).reshape(self.spk_size,
                                                                     config.EMBEDDING_SIZE)
