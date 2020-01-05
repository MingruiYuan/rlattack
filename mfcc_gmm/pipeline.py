import os, random
import numpy as np
from sklearn.mixture import GaussianMixture as gmm
from sklearn.externals import joblib
from data.asvset import mfcc_extractor_antispoofing

def train(cfg, ctime):
    feat_dir = cfg['DATA_DIR']+'features/mfcc_antispoofing/train/'
    if not os.path.exists(feat_dir):
        os.system('mkdir -p '+feat_dir+'bonafide/')
        os.system('mkdir -p '+feat_dir+'spoof/')
        mfcc_extractor_antispoofing(cfg, 'train')

    bnfd_list = os.listdir(feat_dir+'bonafide/')
    spf_list = os.listdir(feat_dir+'spoof/')
    utts_in_use = cfg['UTTS_IN_USE']
    frames_per_utt = cfg['FRAMES_PER_UTT']
    bnfd_feats = []
    spf_feats = []
    random.shuffle(bnfd_list)
    random.shuffle(spf_list)

    for k in range(utts_in_use):
        print('Prepare training features for bonafide data {}/{} {}'.format(str(k+1), str(utts_in_use), bnfd_list[k]))
        bfeat = np.load(feat_dir+'bonafide/'+bnfd_list[k])
        if bfeat.shape[1] > frames_per_utt:
            bfeat = bfeat[:, :frames_per_utt]
        bnfd_feats.append(bfeat)

        print('Prepare training features for spoof data {}/{} {}'.format(str(k+1), str(utts_in_use), spf_list[k]))
        sfeat = np.load(feat_dir+'spoof/'+spf_list[k])
        if sfeat.shape[1] > frames_per_utt:
            sfeat = sfeat[:, :frames_per_utt]
        spf_feats.append(sfeat)

    bnfd_feats = np.hstack(bnfd_feats).T
    spf_feats = np.hstack(spf_feats).T
    print('Bonafide features shape: ', bnfd_feats.shape)
    print('Spoof features shape: ', spf_feats.shape)

    print('Training GMM...')
    bnfd_gmm = gmm(n_components=cfg['N_CPNTS'], max_iter=cfg['MAX_ITER'], init_params='random', verbose=2, verbose_interval=1)
    spf_gmm = gmm(n_components=cfg['N_CPNTS'], max_iter=cfg['MAX_ITER'], init_params='random', verbose=2, verbose_interval=1)
    bnfd_gmm.fit(bnfd_feats)
    spf_gmm.fit(spf_feats)

    print('Training done. Save models.')
    save_dir = cfg['ROOT_DIR'] + 'saved_models/gmm/'
    if not os.path.exists(save_dir):
        os.system('mkdir -p '+save_dir)

    joblib.dump(bnfd_gmm, save_dir+'{}_bonafide.gmm'.format(ctime))
    joblib.dump(spf_gmm, save_dir+'{}_spoof.gmm'.format(ctime))

def validate(cfg, ctime):
    feat_dir = cfg['DATA_DIR']+'features/mfcc_antispoofing/dev/'
    if not os.path.exists(feat_dir):
        os.system('mkdir -p '+feat_dir+'bonafide/')
        os.system('mkdir -p '+feat_dir+'spoof/')
        mfcc_extractor_antispoofing(cfg, 'dev')

def evaluate(cfg, ctime):
    feat_dir = cfg['DATA_DIR']+'features/mfcc_antispoofing/eval/'
    if not os.path.exists(feat_dir):
        os.system('mkdir -p '+feat_dir+'bonafide/')
        os.system('mkdir -p '+feat_dir+'spoof/')
        mfcc_extractor_antispoofing(cfg, 'eval')