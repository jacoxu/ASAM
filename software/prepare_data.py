# <-*- encoding: utf-8 -*->
"""
    pre-process data
"""
import numpy as np
import random
import config
import soundfile as sf
import resampy
import librosa

__author__ = '[jacoxu](https://github.com/jacoxu)'


def get_idx(train_list, valid_list=None, test_list=None):
    spk_set = set()
    audio_path_list = []
    if train_list is not None:
        audio_path_list.append(train_list)
    else:
        raise Exception("Error, train_list should not be None.")
    if valid_list is not None:
        audio_path_list.append(valid_list)
    if test_list is not None:
        audio_path_list.append(test_list)

    for audio_list in audio_path_list:
        file_list = open(audio_list)
        for line in file_list:
            line = line.strip().split()
            if len(line) < 2:
                print 'Wrong audio list file record in the line:', line
                continue
            spk = line[-1]
            spk_set.add(spk)
        file_list.close()
    spk_to_idx = {}
    for spk in spk_set:
        spk_to_idx[spk] = int(spk[-2:])
    idx_to_spk = {}
    for spk, idx in spk_to_idx.iteritems():
        idx_to_spk[idx] = spk
    return spk_to_idx, idx_to_spk


def get_dims(generator):
    inp, out = next(generator)
    inp_fea_len = inp['input_mix_feature'].shape[1]
    inp_fea_dim = inp['input_mix_feature'].shape[-1]
    inp_spec_dim = inp['input_mix_spectrum'].shape[-1]
    inp_spk_len = inp['input_target_spk'].shape[-1]
    out_spec_dim = out['target_clean_spectrum'].shape[-1]
    return inp_fea_len, inp_fea_dim, inp_spec_dim, inp_spk_len, out_spec_dim


def get_feature(audio_list, spk_to_idx, min_mix=2, max_mix=2, batch_size=1):
    """
    :param audio_list: audio file list
        path/to/1st.wav spk1
        path/to/2nd.wav spk2
        path/to/3rd.wav spk1
    :param spk_to_idx: dict, spk1:0, spk2:1, ...
    :param min_mix:
    :param max_mix:
    :param batch_size:
    :return:
    """
    speaker_audios = {}
    batch_input_mix_fea = []
    batch_input_mix_spec = []
    batch_input_spk = []
    batch_input_clean_fea = []
    batch_target_spec = []
    batch_input_len = []
    batch_count = 0
    while True:
        mix_k = np.random.randint(min_mix, max_mix+1)

        if mix_k > len(speaker_audios):
            speaker_audios = {}
            file_list = open(audio_list)
            for line in file_list:
                line = line.strip().split()
                if len(line) != 2:
                    print 'Wrong audio list file record in the line:', line
                    continue
                file_str, spk = line
                if spk not in speaker_audios:
                    speaker_audios[spk] = []
                speaker_audios[spk].append(file_str)
            file_list.close()

            for spk in speaker_audios:
                random.shuffle(speaker_audios[spk])

        wav_mix = None
        target_spk = None
        mix_len = 0
        target_sig = None

        for spk in random.sample(speaker_audios.keys(), mix_k):
            file_str = speaker_audios[spk].pop()
            if not speaker_audios[spk]:
                del(speaker_audios[spk])
            signal, rate = sf.read(file_str)
            if len(signal.shape) > 1:
                signal = signal[:, 0]
            if rate != config.FRAME_RATE:
                signal = resampy.resample(signal, rate, config.FRAME_RATE, filter='kaiser_best')
            signal = list(signal)
            if len(signal) > config.MAX_LEN:
                signal = signal[:config.MAX_LEN]
            if len(signal) > mix_len:
                mix_len = len(signal)

            signal = np.array(signal)
            signal -= np.mean(signal)
            signal /= np.max(np.abs(signal))

            signal = list(signal)

            if config.AUGMENT_DATA:
                random_shift = random.sample(range(len(signal)), 1)[0]
                signal = signal[random_shift:] + signal[:random_shift]

            if len(signal) < config.MAX_LEN:
                signal.extend(np.zeros(config.MAX_LEN - len(signal)))

            signal = np.array(signal)

            if wav_mix is None:
                wav_mix = signal
                target_sig = signal
                target_spk = spk_to_idx[spk]
            else:
                wav_mix = wav_mix + signal

        if config.IS_LOG_SPECTRAL:
            feature_mix = np.log(np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                                config.FRAME_SHIFT,
                                                                                window=config.WINDOWS)))
                                 + np.spacing(1))
        else:
            feature_mix = np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                         config.FRAME_SHIFT,
                                                                         window=config.WINDOWS)))

        spec_mix = np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                  config.FRAME_SHIFT, window=config.WINDOWS)))

        if config.IS_LOG_SPECTRAL:
            feature_inp_clean = np.log(np.transpose(np.abs(librosa.core.spectrum.stft(target_sig, config.FRAME_LENGTH,
                                                                                      config.FRAME_SHIFT,
                                                                                      window=config.WINDOWS)))
                                       + np.spacing(1))
        else:
            feature_inp_clean = np.transpose(np.abs(librosa.core.spectrum.stft(target_sig, config.FRAME_LENGTH,
                                                                               config.FRAME_SHIFT,
                                                                               window=config.WINDOWS)))

        spec_clean = np.transpose(np.abs(librosa.core.spectrum.stft(target_sig, config.FRAME_LENGTH,
                                                                    config.FRAME_SHIFT, window=config.WINDOWS)))


        batch_input_mix_fea.append(feature_mix)
        batch_input_mix_spec.append(spec_mix)
        batch_input_spk.append(target_spk)
        batch_input_clean_fea.append(feature_inp_clean)
        batch_target_spec.append(spec_clean)
        batch_input_len.append(mix_len)
        batch_count += 1

        if batch_count == batch_size:
            # mix_input_fea (batch_size, time_steps, feature_dim)
            mix_input_fea = np.array(batch_input_mix_fea).reshape((batch_size, ) + feature_mix.shape)
            # mix_input_spec (batch_size, time_steps, spectrum_dim)
            mix_input_spec = np.array(batch_input_mix_spec).reshape((batch_size, ) + spec_mix.shape)
            # target_input_spk (batch_size, 1)
            target_input_spk = np.array(batch_input_spk, dtype=np.int32).reshape((batch_size, 1))
            # clean_input_fea (batch_size, time_steps, feature_dim)
            clean_input_fea = np.array(batch_input_clean_fea).reshape((batch_size, ) + feature_inp_clean.shape)
            # clean_target_spec (batch_size, time_steps, spectrum_dim)
            clean_target_spec = np.array(batch_target_spec).reshape((batch_size, ) + spec_clean.shape)

            yield ({'input_mix_feature': mix_input_fea, 'input_mix_spectrum': mix_input_spec,
                    'input_target_spk': target_input_spk, 'input_clean_feature': clean_input_fea},
                   {'target_clean_spectrum': clean_target_spec})
            batch_input_mix_fea = []
            batch_input_mix_spec = []
            batch_input_spk = []
            batch_input_clean_fea = []
            batch_target_spec = []
            batch_input_len = []
            batch_count = 0

if __name__ == "__main__":
    config.init_config()
    spk_to_idx, idx_to_spk = get_idx(config.TRAIN_LIST, config.VALID_LIST, config.TEST_LIST)
    x, y = next(get_feature(config.TRAIN_LIST, spk_to_idx, min_mix=config.MIN_MIX, max_mix=config.MAX_MIX,
                            batch_size=config.BATCH_SIZE))
    print (x['input_mix_feature'].shape)
    print (x['input_mix_spectrum'].shape)
    print (x['input_target_spk'].shape)
    print (x['input_clean_feature'].shape)
    print (y['target_clean_spectrum'].shape)
