function score = get_reward(toolkit_dir, filepath)

clc;
% add required libraries to the path
addpath(genpath(horzcat(toolkit_dir, 'LFCC')));
addpath(genpath(horzcat(toolkit_dir, 'CQCC_v1.0')));
addpath(genpath(horzcat(toolkit_dir, 'GMM')));
addpath(genpath(horzcat(toolkit_dir, 'bosaris_toolkit')));
addpath(genpath(horzcat(toolkit_dir, 'tDCF_v1')));

[x,fs] = audioread(filepath);
[stat, delta, double_delta] = extract_lfcc(x, fs, 20, 512, 20);
feat = [stat delta double_delta]';
% threshold = 0.748917;

load saved_models/gmm_lfcc/ggmm_mu.mat
load saved_models/gmm_lfcc/ggmm_sigma.mat
load saved_models/gmm_lfcc/ggmm_weight.mat
load saved_models/gmm_lfcc/sgmm_mu.mat
load saved_models/gmm_lfcc/sgmm_sigma.mat
load saved_models/gmm_lfcc/sgmm_weight.mat

llk_genuine = mean(compute_llk(feat, genuineGMM_m, genuineGMM_s, genuineGMM_w));
llk_spoof = mean(compute_llk(feat, spoofGMM_m, spoofGMM_s, spoofGMM_w));
score = llk_genuine - llk_spoof;
% reward = score > threshold;