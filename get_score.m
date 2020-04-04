function score = get_score(root_dir, ft_type, filepath)

clc;
% add required libraries to the path
addpath(genpath(horzcat(root_dir, 'GMM_toolbox/', 'LFCC')));
addpath(genpath(horzcat(root_dir, 'GMM_toolbox/', 'CQCC_v1.0')));
addpath(genpath(horzcat(root_dir, 'GMM_toolbox/', 'GMM')));
addpath(genpath(horzcat(root_dir, 'GMM_toolbox/', 'bosaris_toolkit')));
addpath(genpath(horzcat(root_dir, 'GMM_toolbox/', 'tDCF_v1')));

[x,fs] = audioread(filepath);

if strcmp(ft_type, 'LFCC')
	[stat, delta, double_delta] = extract_lfcc(x, fs, 20, 512, 20);
    feat = [stat delta double_delta]';
	load saved_models/gmm_antispoofing/ggmm_mu_lfcc.mat
	load saved_models/gmm_antispoofing/ggmm_sigma_lfcc.mat
	load saved_models/gmm_antispoofing/ggmm_weight_lfcc.mat
	load saved_models/gmm_antispoofing/sgmm_mu_lfcc.mat
	load saved_models/gmm_antispoofing/sgmm_sigma_lfcc.mat
	load saved_models/gmm_antispoofing/sgmm_weight_lfcc.mat
elseif strcmp(ft_type, 'CQCC')
	feat = cqcc(x, fs, 96, fs/2, fs/2^10, 16, 29, 'ZsdD');
	load saved_models/gmm_antispoofing/ggmm_mu_cqcc.mat
	load saved_models/gmm_antispoofing/ggmm_sigma_cqcc.mat
	load saved_models/gmm_antispoofing/ggmm_weight_cqcc.mat
	load saved_models/gmm_antispoofing/sgmm_mu_cqcc.mat
	load saved_models/gmm_antispoofing/sgmm_sigma_cqcc.mat
	load saved_models/gmm_antispoofing/sgmm_weight_cqcc.mat
end

llk_genuine = mean(compute_llk(feat, genuineGMM_m, genuineGMM_s, genuineGMM_w));
llk_spoof = mean(compute_llk(feat, spoofGMM_m, spoofGMM_s, spoofGMM_w));
score = llk_genuine - llk_spoof;