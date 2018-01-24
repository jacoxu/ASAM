function Parms =  BSS_EVAL...
  (wav_truth_signal, wav_truth_noise, wav_pred_signal, wav_mix)
% Evaluate performance using BSS Eval 2.0
addpath(genpath('./matlab/'))

%% evaluate
sep = wav_pred_signal;
orig = [wav_truth_noise; wav_truth_signal];

[e1,e2,e3] = bss_decomp_gain( sep, 2, orig);
[sdr,sir,sar] = bss_crit( e1, e2, e3);



[e1,e2,e3] = bss_decomp_gain( wav_mix, 1, wav_truth_signal);
[sdr_,sir_,sar_] = bss_crit( e1, e2, e3);


Parms.SDR=sdr;
Parms.SIR=sir;
Parms.SAR=sar;
Parms.NSDR=Parms.SDR-sdr_;
