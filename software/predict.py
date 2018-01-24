# -*- coding: utf-8 -*-

import matlab
import soundfile as sf
import numpy as np
import config
import resampy
import librosa
import time

__author__ = '[jacoxu](https://github.com/jacoxu)'


def compute_batch_clean_fea(batch_clean_wav):
    batch_input_clean_fea = []
    feature_inp_clean_shape = None
    tmp_batch_maxlen = 0
    # find the max length
    for tar_supp_sig in batch_clean_wav:
        if len(tar_supp_sig) > tmp_batch_maxlen:
            tmp_batch_maxlen = len(tar_supp_sig)
    # pad with zero
    for signal in batch_clean_wav:
        signal -= np.mean(signal)
        signal /= np.max(np.abs(signal))

        signal = list(signal)
        if len(signal) < tmp_batch_maxlen:
            signal.extend(np.zeros(tmp_batch_maxlen - len(signal)))

        signal = np.array(signal)

        if config.IS_LOG_SPECTRAL:
            feature_inp_clean = np.log(np.transpose(np.abs(librosa.core.spectrum.stft(signal, config.FRAME_LENGTH,
                                                                                      config.FRAME_SHIFT,
                                                                                      window=config.WINDOWS)))
                                       + np.spacing(1))
        else:
            feature_inp_clean = np.transpose(np.abs(librosa.core.spectrum.stft(signal, config.FRAME_LENGTH,
                                                                               config.FRAME_SHIFT,
                                                                               window=config.WINDOWS)))
        feature_inp_clean_shape = feature_inp_clean.shape
        batch_input_clean_fea.append(feature_inp_clean)

    return batch_input_clean_fea, feature_inp_clean_shape


def eval_separation(model, audio_list, valid_test, epoch_num, log_file, spk_to_idx, batch_size=1, spk_num=2
                    , unk_spk=False, supp_time=1, add_bgd_noise=False):
    if unk_spk:
        batch_size = 1
    if spk_num < 2:
        spk_num = 2
    batch_input_mix_fea = []
    batch_input_mix_spec = []
    batch_input_spk = []
    batch_input_len = []
    batch_mix_spec = []
    batch_mix_wav = []
    batch_target_wav = []
    batch_noise_wav = []
    batch_clean_wav = []
    batch_count = 0

    batch_sdr_0 = []
    batch_sir_0 = []
    batch_sar_0 = []
    batch_nsdr_0 = []
    batch_sdr = []
    batch_sir = []
    batch_sar = []
    batch_nsdr = []

    file_list_len = 0
    file_list = open(audio_list)
    for line in file_list:
        file_list_len += 1
    file_list.close()
    file_list = open(audio_list)
    time_start = time.time()
    for line_idx, line in enumerate(file_list):
        line = line.strip().split()
        if len(line) < 2:
            raise Exception('Wrong audio list file record in the line:', ''.join(line))
        file_tar_sounds_str = None
        if not unk_spk:
            # if not test unk_spk
            file_tar_str, file_bg_str, tar_spk_str = line
            file_bg_str = file_bg_str.strip().split(',')
        else:
            # if test unk_spk
            file_tar_str, file_bg_str, tar_spk_str, file_tar_sounds_str = line
            file_bg_str = [file_bg_str]
        wav_mix = None
        target_spk = None
        mix_len = 0
        target_sig = None
        noise_sig = None
        tar_supp_sig = None

        for file_str in ([file_tar_str]+file_bg_str)[:spk_num]:
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
            if len(signal) < config.MAX_LEN:
                signal.extend(np.zeros(config.MAX_LEN - len(signal)))
            signal = np.array(signal)

            if wav_mix is None:
                wav_mix = signal
                target_sig = signal
                if not unk_spk:
                    tar_supp_sig = signal
                    batch_clean_wav.append(tar_supp_sig)
                    target_spk = spk_to_idx[tar_spk_str]
                else:
                    # idx of unk_spk: 0
                    target_spk = 0
            else:
                wav_mix = wav_mix + signal
                if noise_sig is None:
                    noise_sig = signal
                else:
                    noise_sig = noise_sig + signal

        if add_bgd_noise:
            bg_noise = config.BGD_NOISE_WAV[:config.MAX_LEN]
            bg_noise -= np.mean(bg_noise)
            bg_noise /= np.max(np.abs(bg_noise))

            wav_mix = wav_mix + bg_noise
            noise_sig = noise_sig + bg_noise

        if unk_spk:
            tmp_unk_spk_supp = 0
            for file_str in file_tar_sounds_str.strip().split(','):
                tmp_unk_spk_supp += 1
                if tmp_unk_spk_supp > config.UNK_SPK_SUPP:
                    break
                signal, rate = sf.read(file_str)
                if len(signal.shape) > 1:
                    signal = signal[:, 0]
                if rate != config.FRAME_RATE:
                    signal = resampy.resample(signal, rate, config.FRAME_RATE, filter='kaiser_best')
                signal = list(signal)
                if tar_supp_sig is None:
                    tar_supp_sig = signal
                else:
                    tar_supp_sig = tar_supp_sig + signal
            if len(tar_supp_sig) < supp_time*config.FRAME_RATE:
                raise Exception('the supp_time is too greater than the target supplemental sounds!')
            batch_clean_wav.append(tar_supp_sig[:int(supp_time * config.FRAME_RATE)])

        if config.IS_LOG_SPECTRAL:
            feature_mix = np.log(np.abs(np.transpose(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                                config.FRAME_SHIFT,
                                                                                window=config.WINDOWS)))
                                 + np.spacing(1))
        else:
            feature_mix = np.abs(np.transpose(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                         config.FRAME_SHIFT,
                                                                         window=config.WINDOWS)))

        spec_mix = np.transpose(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                           config.FRAME_SHIFT,
                                                           window=config.WINDOWS))

        batch_input_mix_fea.append(feature_mix)
        batch_input_mix_spec.append(np.abs(spec_mix))
        batch_input_spk.append(target_spk)
        batch_input_len.append(mix_len)
        batch_mix_spec.append(spec_mix)
        batch_mix_wav.append(wav_mix)
        batch_target_wav.append(target_sig)
        batch_noise_wav.append(noise_sig)

        batch_count += 1

        if (batch_count == batch_size) or (line_idx == (file_list_len-1)):
            # mix_input_fea (batch_size, time_steps, feature_dim)
            _tmp_batch_size = len(batch_input_mix_fea)
            mix_input_fea = np.array(batch_input_mix_fea).reshape((_tmp_batch_size, ) + feature_mix.shape)
            # mix_input_spec (batch_size, time_steps, spectrum_dim)
            mix_input_spec = np.array(batch_input_mix_spec).reshape((_tmp_batch_size, ) + spec_mix.shape)
            # bg_input_mask = np.array(batch_input_silence_mask).reshape((_tmp_batch_size, ) + spec_mix.shape)
            # target_input_spk (batch_size, 1)
            target_input_spk = np.array(batch_input_spk).reshape((_tmp_batch_size, 1))
            # clean_input_fea (batch_size, time_steps, feature_dim)
            batch_input_clean_fea, inp_clean_shape = compute_batch_clean_fea(batch_clean_wav)
            clean_input_fea = np.array(batch_input_clean_fea).reshape((batch_size, ) + inp_clean_shape)
            if not unk_spk:
                clean_input_fea = np.log(np.zeros_like(clean_input_fea)+np.spacing(1))
            target_pred = model.predict({'input_mix_feature': mix_input_fea, 'input_mix_spectrum': mix_input_spec,
                                         'input_target_spk': target_input_spk, 'input_clean_feature': clean_input_fea})
            batch_idx = 0
            for _pred_output in list(target_pred):
                _mix_spec = batch_mix_spec[batch_idx]
                phase_mix = np.angle(_mix_spec)
                _pred_spec = _pred_output * np.exp(1j * phase_mix)
                _pred_wav = librosa.core.spectrum.istft(np.transpose(_pred_spec), config.FRAME_SHIFT,
                                                        window=config.WINDOWS)
                _target_wav = batch_target_wav[batch_idx]
                min_len = np.min((len(_target_wav), len(_pred_wav), batch_input_len[batch_idx]))
                _pred_wav = _pred_wav[:min_len]
                batch_target_wav[batch_idx] = _target_wav[:min_len]
                batch_noise_wav[batch_idx] = batch_noise_wav[batch_idx][:min_len]
                batch_mix_wav[batch_idx] = batch_mix_wav[batch_idx][:min_len]

                mix_wav = matlab.double(batch_mix_wav[batch_idx].tolist())
                target_wav = matlab.double(batch_target_wav[batch_idx].tolist())
                noise_wav = matlab.double(batch_noise_wav[batch_idx].tolist())
                pred_wav = matlab.double(_pred_wav.tolist())
                if epoch_num == 0:
                    # BSS_EVAL (truth_signal, truth_noise, pred_signal, mix)
                    bss_eval_resuts = config.MAT_ENG.BSS_EVAL(target_wav, noise_wav, mix_wav, mix_wav)
                    batch_sdr_0.append(bss_eval_resuts['SDR'])
                    batch_sir_0.append(bss_eval_resuts['SIR'])
                    batch_sar_0.append(bss_eval_resuts['SAR'])
                    batch_nsdr_0.append(bss_eval_resuts['NSDR'])
                if (line_idx < _tmp_batch_size) and (batch_idx == 0):
                    sf.write(config.TMP_PRED_WAV_FOLDER + '/test_pred_%s_ep%04d_bs%04d_idx%03d' %
                             (config.DATASET, (epoch_num+1), 1, (batch_idx+1)) +
                             '.wav', _pred_wav, config.FRAME_RATE)

                # BSS_EVAL (truth_signal, truth_noise, pred_signal, mix)
                bss_eval_resuts = config.MAT_ENG.BSS_EVAL(target_wav, noise_wav, pred_wav, mix_wav)
                batch_sdr.append(bss_eval_resuts['SDR'])
                batch_sir.append(bss_eval_resuts['SIR'])
                batch_sar.append(bss_eval_resuts['SAR'])
                batch_nsdr.append(bss_eval_resuts['NSDR'])
                batch_idx += 1

            time_end = time.time()
            sdr = np.float(np.mean(batch_sdr))
            sir = np.float(np.mean(batch_sir))
            sar = np.float(np.mean(batch_sar))
            nsdr = np.float(np.mean(batch_nsdr))
            print '\rCurrent predict:' + str(line_idx+1) + ' of ' + audio_list + \
                ' and cost time: %.4f sec. - GSDR:%f, GSIR:%f, GSAR:%f, GNSDR:%f' % ((time_end - time_start),
                                                                                     sdr, sir, sar, nsdr),
            if (line_idx+1) % 200 == 0:
                log_file.write('Have evaluated %05d mixture wavs, and cost time: %.4f sec\n'
                               % ((line_idx+1), (time_end - time_start)))
                log_file.flush()
            batch_input_mix_fea = []
            batch_input_mix_spec = []
            batch_input_spk = []
            batch_input_len = []
            batch_mix_spec = []
            batch_mix_wav = []
            batch_target_wav = []
            batch_noise_wav = []
            batch_clean_wav = []
            batch_count = 0

    if epoch_num == 0:
        sdr_0 = np.float(np.mean(batch_sdr_0))
        sir_0 = np.float(np.mean(batch_sir_0))
        sar_0 = np.float(np.mean(batch_sar_0))
        nsdr_0 = np.float(np.mean(batch_nsdr_0))
        print '\n[Epoch-%s: %d] - GSDR:%f, GSIR:%f, GSAR:%f, GNSDR:%f' % \
              (valid_test, epoch_num, sdr_0, sir_0, sar_0, nsdr_0)
        log_file.write('[Epoch-%s: %d] - GSDR:%f, GSIR:%f, GSAR:%f, GNSDR:%f\n' %
                       (valid_test, epoch_num, sdr_0, sir_0, sar_0, nsdr_0))
        log_file.flush()

    sdr = np.float(np.mean(batch_sdr))
    sir = np.float(np.mean(batch_sir))
    sar = np.float(np.mean(batch_sar))
    nsdr = np.float(np.mean(batch_nsdr))
    if epoch_num == 0:
        print '[Epoch-%s: %d] - GSDR:%f, GSIR:%f, GSAR:%f, GNSDR:%f' % \
              (valid_test, epoch_num+1, sdr, sir, sar, nsdr)
    else:
        print '\n[Epoch-%s: %d] - GSDR:%f, GSIR:%f, GSAR:%f, GNSDR:%f' % \
              (valid_test, epoch_num+1, sdr, sir, sar, nsdr)
    log_file.write('[Epoch-%s: %d] - GSDR:%f, GSIR:%f, GSAR:%f, GNSDR:%f\n' %
                   (valid_test, epoch_num+1, sdr, sir, sar, nsdr))
    log_file.flush()
    file_list.close()


def eval_loss(model, audio_list, valid_test, epoch_num, log_file, spk_to_idx, batch_size=1, unk_spk=False):
    if unk_spk:
        batch_size = 1
    batch_input_mix_fea = []
    batch_input_mix_spec = []
    batch_input_spk = []
    batch_target_spec = []
    batch_input_len = []
    batch_clean_wav = []
    batch_count = 0
    mse_loss = 0

    file_list_len = 0
    file_list = open(audio_list)
    for line in file_list:
        file_list_len += 1
    file_list.close()
    file_list = open(audio_list)
    time_start = time.time()
    for line_idx, line in enumerate(file_list):
        line = line.strip().split()
        if len(line) < 2:
            raise Exception('Wrong audio list file record in the line:', ''.join(line))
        file_tar_sounds_str = None
        if not unk_spk:
            file_tar_str, file_bg_str, tar_spk_str = line
            file_bg_str = file_bg_str.strip().split(',')[0]
        else:
            file_tar_str, file_bg_str, tar_spk_str, file_tar_sounds_str = line

        wav_mix = None
        target_spk = None
        mix_len = 0
        target_sig = None
        tar_supp_sig = None

        for file_str in [file_tar_str, file_bg_str]:
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
            if len(signal) < config.MAX_LEN:
                signal.extend(np.zeros(config.MAX_LEN - len(signal)))
            signal = np.array(signal)

            if wav_mix is None:
                wav_mix = signal
                target_sig = signal
                if not unk_spk:
                    tar_supp_sig = signal
                    batch_clean_wav.append(tar_supp_sig)
                    target_spk = spk_to_idx[tar_spk_str]
                else:
                    target_spk = 0
            else:
                wav_mix = wav_mix + signal

        if unk_spk:
            tmp_unk_spk_supp = 0
            for file_str in file_tar_sounds_str.strip().split(','):
                tmp_unk_spk_supp += 1
                if tmp_unk_spk_supp > config.UNK_SPK_SUPP:
                    break
                signal, rate = sf.read(file_str)
                if len(signal.shape) > 1:
                    signal = signal[:, 0]
                if rate != config.FRAME_RATE:
                    signal = resampy.resample(signal, rate, config.FRAME_RATE, filter='kaiser_best')
                if tar_supp_sig is None:
                    tar_supp_sig = signal
                else:
                    tar_supp_sig = tar_supp_sig + signal
            batch_clean_wav.append(tar_supp_sig)

        if config.IS_LOG_SPECTRAL:
            feature_mix = np.log(np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                                config.FRAME_SHIFT,
                                                                                window=config.WINDOWS)))
                                 + np.spacing(1))
        else:
            feature_mix = np.transpose(np.abs(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                                         config.FRAME_SHIFT,
                                                                         window=config.WINDOWS)))

        spec_mix = np.transpose(librosa.core.spectrum.stft(wav_mix, config.FRAME_LENGTH,
                                                           config.FRAME_SHIFT,
                                                           window=config.WINDOWS))

        spec_clean = np.transpose(np.abs(librosa.core.spectrum.stft(target_sig, config.FRAME_LENGTH,
                                                                    config.FRAME_SHIFT, window=config.WINDOWS)))

        batch_input_mix_fea.append(feature_mix)
        batch_input_mix_spec.append(np.abs(spec_mix))
        batch_input_spk.append(target_spk)
        batch_target_spec.append(spec_clean)
        batch_input_len.append(mix_len)

        batch_count += 1

        if (batch_count == batch_size) or (line_idx == (file_list_len-1)):
            # mix_input_fea (batch_size, time_steps, feature_dim)
            _tmp_batch_size = len(batch_input_mix_fea)
            mix_input_fea = np.array(batch_input_mix_fea).reshape((_tmp_batch_size, ) + feature_mix.shape)
            # mix_input_spec (batch_size, time_steps, spectrum_dim)
            mix_input_spec = np.array(batch_input_mix_spec).reshape((_tmp_batch_size, ) + spec_mix.shape)
            # target_input_spk (batch_size, 1)
            target_input_spk = np.array(batch_input_spk).reshape((_tmp_batch_size, 1))
            # clean_input_fea (batch_size, time_steps, feature_dim)
            batch_input_clean_fea, inp_clean_shape = compute_batch_clean_fea(batch_clean_wav)
            clean_input_fea = np.array(batch_input_clean_fea).reshape((_tmp_batch_size, ) + inp_clean_shape)
            # clean_target_spec (batch_size, time_steps, spectrum_dim)
            clean_target_spec = np.array(batch_target_spec).reshape((_tmp_batch_size, ) + spec_clean.shape)

            if not unk_spk:
                clean_input_fea = np.zeros_like(clean_input_fea)

            mse_loss += model.evaluate({'input_mix_feature': mix_input_fea, 'input_mix_spectrum': mix_input_spec,
                                        'input_target_spk': target_input_spk, 'input_clean_feature': clean_input_fea},
                                       {'target_clean_spectrum': clean_target_spec}, batch_size=_tmp_batch_size,
                                       verbose=0)

            time_end = time.time()
            print '\rCurrent evaluate:' + str(line_idx+1) + ' of ' + audio_list + \
                ' and cost time: %.4f sec.' % (time_end - time_start),

            batch_input_mix_fea = []
            batch_input_mix_spec = []
            batch_input_spk = []
            batch_target_spec = []
            batch_input_len = []
            batch_clean_wav = []
            batch_count = 0

    print '\n[Epoch-%s: %d] - MSE Loss:%f' % \
          (valid_test, epoch_num+1, mse_loss)
    log_file.write('[Epoch-%s: %d] - MSE Loss:%f\n' %
                   (valid_test, epoch_num+1, mse_loss))
    log_file.flush()
    file_list.close()
    return mse_loss
