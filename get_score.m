function score = get_score(toolkit_dir, ft_type, filepath)

clc;
% add required libraries to the path
addpath(genpath(horzcat(toolkit_dir, 'LFCC')));
addpath(genpath(horzcat(toolkit_dir, 'CQCC_v1.0')));
addpath(genpath(horzcat(toolkit_dir, 'GMM')));
addpath(genpath(horzcat(toolkit_dir, 'bosaris_toolkit')));
addpath(genpath(horzcat(toolkit_dir, 'tDCF_v1')));

[x,fs] = audioread(filepath);

if strcmp(ft_type, 'LFCC')
	[stat, delta, double_delta] = extract_lfcc(x, fs, 20, 512, 20);
    feat = [stat delta double_delta]';
	load saved_models/gmm_lfcc/ggmm_mu.mat
	load saved_models/gmm_lfcc/ggmm_sigma.mat
	load saved_models/gmm_lfcc/ggmm_weight.mat
	load saved_models/gmm_lfcc/sgmm_mu.mat
	load saved_models/gmm_lfcc/sgmm_sigma.mat
	load saved_models/gmm_lfcc/sgmm_weight.mat
elseif strcmp(ft_type, 'CQCC')
	feat = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
	load saved_models/gmm_lfcc/ggmm_mu_cqcc.mat
	load saved_models/gmm_lfcc/ggmm_sigma_cqcc.mat
	load saved_models/gmm_lfcc/ggmm_weight_cqcc.mat
	load saved_models/gmm_lfcc/sgmm_mu_cqcc.mat
	load saved_models/gmm_lfcc/sgmm_sigma_cqcc.mat
	load saved_models/gmm_lfcc/sgmm_weight_cqcc.mat
end

llk_genuine = mean(compute_llk(feat, genuineGMM_m, genuineGMM_s, genuineGMM_w));
llk_spoof = mean(compute_llk(feat, spoofGMM_m, spoofGMM_s, spoofGMM_w));
score = llk_genuine - llk_spoof;
% reward = score > threshold;