# <-*- encoding:utf8 -*->

"""
    Configuration Profile
"""
import matlab.engine
import time
import ConfigParser
import soundfile as sf
import resampy
import numpy as np
from scipy import signal

__author__ = 'https://github.com/jacoxu/ASAM'


# Has loaded the configuration
HAS_INIT_CONFIG = False
MAT_ENG = []
# External configuration file
CONFIG_FILE = './config.cfg'
# Record log into this file, such as ASAM_output.log_20180101_110305
LOG_FILE_PRE = './ASAM_output.log'
# mode=1 for tranining and evaluation, 2 for only evaluation
MODE = 1
# Dataset: THCHS-30 or WSJ0
DATASET = 'WSJ0'
# Training file path list
TRAIN_LIST = './train_wavlist_'+DATASET
# Valid file path list
VALID_LIST = './valid_wavlist_'+DATASET
# Test file path list
TEST_LIST = './test_wavlist_'+DATASET
# Unkown speech file path list
UNK_LIST = './unk_wavlist_'+DATASET
# Hidden units of DNN/RNN
HIDDEN_UNITS = 16
# Layer number of DNN/RNN
NUM_LAYERS = 1
# Size of embedding (please set a even integer)
EMBEDDING_SIZE = 20
# Whether augment training data
AUGMENT_DATA = False
# set the max epoch of training
MAX_EPOCH = 5
# Epoch size, for example 100 batches per epoch
EPOCH_SIZE = 20
# Batch size, for example 32 samples per batch
BATCH_SIZE = 2
# Batch size in evaluation phase
BATCH_SIZE_EVAL = 10
# feature frame rate
FRAME_RATE = 8000
# Frame length (Sample number)
FRAME_LENGTH = int(0.032 * FRAME_RATE)
# Frame shift (Sample number)
FRAME_SHIFT = int(0.016 * FRAME_RATE)
# Whether shuffle batch data
SHUFFLE_BATCH = True
# Minimum number of mixed speakers for training
MIN_MIX = 2
# Maximum number of mixed speakers for training
MAX_MIX = 2
# Max length (Second) of training/valid/test data
MAX_LEN = 5
# Frame length for window function
WINDOWS = FRAME_LENGTH
# Directory for saving predicted speech
TMP_PRED_WAV_FOLDER = '_tmp_pred_wavs'
# Directory for saving training weights
TMP_WEIGHT_FOLDER = '_tmp_weights'
# Unknown speaker test. False for top-down attention, and True for bottom-up attention
UNK_SPK = False
# Max supplemental number of stimulus speech
UNK_SPK_SUPP = 10
START_EALY_STOP = 0
# Log Spectral
IS_LOG_SPECTRAL = False
# Add background noise (Street)
ADD_BGD_NOISE = False
BGD_NOISE_WAV = None
BGD_NOISE_FILE = './../../dataset/BGD_150203_010_STR.CH1.wav'


def load_bgd_wav(file_path):
    signal, rate = sf.read(file_path)  # signal: sample values，rate: sample rate
    if len(signal.shape) > 1:
        signal = signal[:, 0]
    if rate != FRAME_RATE:
        # up-sample or down-sample for predefined sample rate
        signal = resampy.resample(signal, rate, FRAME_RATE, filter='kaiser_fast')
    return signal


def update_max_len(file_path_list, max_len):
    tmp_max_len = 0
    # Update the max length based on the given dataset
    signal_set = set()
    for file_path in file_path_list:
        file_list = open(file_path)
        for line in file_list:
            line = line.strip().split()
            if len(line) < 2:
                print 'Wrong audio list file record in the line:', line
                continue
            file_str = line[0]
            if file_str in signal_set:
                continue
            signal_set.add(file_str)
            signal, rate = sf.read(file_str)  # signal: sample values，rate: sample rate
            if len(signal.shape) > 1:
                signal = signal[:, 0]
            if rate != FRAME_RATE:
                # up-sample or down-sample for predefined sample rate
                signal = resampy.resample(signal, rate, FRAME_RATE, filter='kaiser_fast')
            if len(signal) > tmp_max_len:
                tmp_max_len = len(signal)
        file_list.close()
    if tmp_max_len < max_len:
        max_len = tmp_max_len
    return max_len


def init_config():
    global HAS_INIT_CONFIG
    if HAS_INIT_CONFIG:
        raise Exception("This config has been initialized")
    # open matlab engine for BSS evaluation
    global MAT_ENG
    MAT_ENG = matlab.engine.start_matlab()
    print 'has opened the matlab engine'
    _config = ConfigParser.ConfigParser()
    cfg_file = open(CONFIG_FILE, 'r')
    _config.readfp(cfg_file)
    global LOG_FILE_PRE
    LOG_FILE_PRE = _config.get('cfg', 'LOG_FILE_PRE').strip()
    global MODE
    MODE = eval(_config.get('cfg', 'MODE'))
    global DATASET
    DATASET = _config.get('cfg', 'DATASET')
    global TRAIN_LIST
    TRAIN_LIST = _config.get('cfg', 'TRAIN_LIST') + DATASET
    global VALID_LIST
    VALID_LIST = _config.get('cfg', 'VALID_LIST') + DATASET
    global TEST_LIST
    TEST_LIST = _config.get('cfg', 'TEST_LIST') + DATASET
    global UNK_LIST
    UNK_LIST = _config.get('cfg', 'UNK_LIST') + DATASET
    global HIDDEN_UNITS
    HIDDEN_UNITS = eval(_config.get('cfg', 'HIDDEN_UNITS'))
    global NUM_LAYERS
    NUM_LAYERS = eval(_config.get('cfg', 'NUM_LAYERS'))
    global EMBEDDING_SIZE
    EMBEDDING_SIZE = eval(_config.get('cfg', 'EMBEDDING_SIZE'))
    if EMBEDDING_SIZE % 2 != 0:
        raise Exception('Embedding size should be even integer.')
    global AUGMENT_DATA
    AUGMENT_DATA = eval(_config.get('cfg', 'AUGMENT_DATA'))
    global MAX_EPOCH
    MAX_EPOCH = eval(_config.get('cfg', 'MAX_EPOCH'))
    global EPOCH_SIZE
    EPOCH_SIZE = eval(_config.get('cfg', 'EPOCH_SIZE'))
    global BATCH_SIZE
    BATCH_SIZE = eval(_config.get('cfg', 'BATCH_SIZE'))
    global BATCH_SIZE_EVAL
    BATCH_SIZE_EVAL = eval(_config.get('cfg', 'BATCH_SIZE_EVAL'))
    global FRAME_RATE
    FRAME_RATE = eval(_config.get('cfg', 'FRAME_RATE'))
    global FRAME_LENGTH
    FRAME_LENGTH = int(eval(_config.get('cfg', 'FRAME_LENGTH')) * FRAME_RATE)
    global FRAME_SHIFT
    FRAME_SHIFT = int(eval(_config.get('cfg', 'FRAME_SHIFT')) * FRAME_RATE)
    global SHUFFLE_BATCH
    SHUFFLE_BATCH = eval(_config.get('cfg', 'SHUFFLE_BATCH'))
    global MIN_MIX
    MIN_MIX = eval(_config.get('cfg', 'MIN_MIX'))
    global MAX_MIX
    MAX_MIX = eval(_config.get('cfg', 'MAX_MIX'))
    global MAX_LEN
    MAX_LEN = int(eval(_config.get('cfg', 'MAX_LEN')) * FRAME_RATE)
    MAX_LEN = update_max_len([TRAIN_LIST, VALID_LIST, TEST_LIST, UNK_LIST], MAX_LEN)
    global WINDOWS
    win_size = FRAME_LENGTH
    # sine window
    WINDOWS = [np.sin(x_i*np.pi/win_size) for x_i in range(win_size)]
    # square root of hanning window
    # WINDOWS = np.sqrt(signal.get_window('hann', win_size))
    # hanning window
    # WINDOWS = signal.get_window('hann', win_size)
    global TMP_PRED_WAV_FOLDER
    TMP_PRED_WAV_FOLDER = _config.get('cfg', 'TMP_PRED_WAV_FOLDER').strip()
    global TMP_WEIGHT_FOLDER
    TMP_WEIGHT_FOLDER = _config.get('cfg', 'TMP_WEIGHT_FOLDER').strip()
    global UNK_SPK
    UNK_SPK = eval(_config.get('cfg', 'UNK_SPK').strip())
    global UNK_SPK_SUPP
    UNK_SPK_SUPP = eval(_config.get('cfg', 'UNK_SPK_SUPP'))

    global ADD_BGD_NOISE
    ADD_BGD_NOISE = eval(_config.get('cfg', 'ADD_BGD_NOISE').strip())
    if ADD_BGD_NOISE:
        global BGD_NOISE_WAV
        print 'Load background (Street) noise wav...'
        BGD_NOISE_WAV = load_bgd_wav(BGD_NOISE_FILE)
    cfg_file.close()


def log_config(_log_file):
    _log_file.write('*' * 80 + '\n')
    _log_file.write('Current time:' + time.strftime('%Y-%m-%d %H:%M:%S') + '\n')
    _log_file.write('MODE:' + str(MODE) + '\n')
    _log_file.write('DATASET:' + str(DATASET) + '\n')
    _log_file.write('TRAIN_LIST:' + str(TRAIN_LIST) + '\n')
    _log_file.write('VALID_LIST:' + str(VALID_LIST) + '\n')
    _log_file.write('TEST_LIST:' + str(TEST_LIST) + '\n')
    _log_file.write('UNK_LIST:' + str(UNK_LIST) + '\n')
    _log_file.write('HIDDEN_UNITS:' + str(HIDDEN_UNITS) + '\n')
    _log_file.write('NUM_LAYERS:' + str(NUM_LAYERS) + '\n')
    _log_file.write('EMBEDDING_SIZE:' + str(EMBEDDING_SIZE) + '\n')
    _log_file.write('AUGMENT_DATA:' + str(AUGMENT_DATA) + '\n')
    _log_file.write('MAX_EPOCH:' + str(MAX_EPOCH) + '\n')
    _log_file.write('EPOCH_SIZE:' + str(EPOCH_SIZE) + '\n')
    _log_file.write('BATCH_SIZE:' + str(BATCH_SIZE) + '\n')
    _log_file.write('BATCH_SIZE_EVAL:' + str(BATCH_SIZE_EVAL) + '\n')
    _log_file.write('FRAME_RATE:' + str(FRAME_RATE) + '\n')
    _log_file.write('FRAME_LENGTH:' + str(FRAME_LENGTH) + '\n')
    _log_file.write('FRAME_SHIFT:' + str(FRAME_SHIFT) + '\n')
    _log_file.write('SHUFFLE_BATCH:' + str(SHUFFLE_BATCH) + '\n')
    _log_file.write('MIN_MIX:' + str(MIN_MIX) + '\n')
    _log_file.write('MAX_MIX:' + str(MAX_MIX) + '\n')
    _log_file.write('MAX_LEN:' + str(MAX_LEN) + '\n')
    _log_file.write('TMP_PRED_WAV_FOLDER:' + str(TMP_PRED_WAV_FOLDER) + '\n')
    _log_file.write('TMP_WEIGHT_FOLDER:' + str(TMP_WEIGHT_FOLDER) + '\n')
    _log_file.write('UNK_SPK:' + str(UNK_SPK) + '\n')
    _log_file.write('UNK_SPK_SUPP:' + str(UNK_SPK_SUPP) + '\n')

    if ADD_BGD_NOISE:
        _log_file.write('Load background (Street) noise wav...\n')
    _log_file.write('*' * 80 + '\n')
    _log_file.flush()
