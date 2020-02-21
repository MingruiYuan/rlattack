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
    #utts_in_use = cfg['UTTS_IN_USE']
    #frames_per_utt = cfg['FRAMES_PER_UTT']
    #bnfd_feats = []
    #spf_feats = []
    all_feats = []
    random.shuffle(bnfd_list)
    random.shuffle(spf_list)

    # for k in range(utts_in_use):
    for k in range(len(bnfd_list)):
        print('Prepare training features for bonafide data {}/{} {}'.format(str(k+1), str(len(bnfd_list)), bnfd_list[k]))
        bfeat = np.load(feat_dir+'bonafide/'+bnfd_list[k])
        # if bfeat.shape[1] > frames_per_utt:
            # bfeat = bfeat[:, :frames_per_utt]
        #bnfd_feats.append(bfeat)
        all_feats.append(bfeat)

    all_feats = np.hstack(all_feats)
    bnfd_gmm = gmm(n_components=cfg['N_CPNTS'], max_iter=cfg['MAX_ITER'], init_params='random', verbose=2, verbose_interval=1)
    bnfd_gmm.fit(all_feats.T)
    all_feats = []

    for k in range(len(spf_list)):
        print('Prepare training features for spoof data {}/{} {}'.format(str(k+1), str(len(spf_list)), spf_list[k]))
        sfeat = np.load(feat_dir+'spoof/'+spf_list[k])
        # if sfeat.shape[1] > frames_per_utt:
            # sfeat = sfeat[:, :frames_per_utt]
        #spf_feats.append(sfeat)
        all_feats.append(sfeat)

    all_feats = np.hstack(all_feats)
    spf_gmm = gmm(n_components=cfg['N_CPNTS'], max_iter=cfg['MAX_ITER'], init_params='random', verbose=2, verbose_interval=1)
    spf_gmm.fit(all_feats.T)

    #bnfd_feats = np.hstack(bnfd_feats).T
    #spf_feats = np.hstack(spf_feats).T
    #print('Bonafide features shape: ', bnfd_feats.shape)
    #print('Spoof features shape: ', spf_feats.shape)

    #print('Training GMM...')
    #bnfd_gmm = gmm(n_components=cfg['N_CPNTS'], max_iter=2*cfg['MAX_ITER'], init_params='random', verbose=2, verbose_interval=1)
    #spf_gmm = gmm(n_components=cfg['N_CPNTS'], max_iter=4*cfg['MAX_ITER'], init_params='random', verbose=2, verbose_interval=1)
    #bnfd_gmm.fit(bnfd_feats)
    #spf_gmm.fit(spf_feats)

    print('Training done. Save models.')
    save_dir = cfg['ROOT_DIR'] + 'saved_models/gmm/'
    if not os.path.exists(save_dir):
        os.system('mkdir -p '+save_dir)

    joblib.dump(bnfd_gmm, save_dir+'{}_bonafide.gmm'.format(ctime))
    joblib.dump(spf_gmm, save_dir+'{}_spoof.gmm'.format(ctime))

def train_warmstart(cfg, ctime):
    feat_dir = cfg['DATA_DIR']+'features/mfcc_antispoofing/train/'
    if not os.path.exists(feat_dir):
        os.system('mkdir -p '+feat_dir+'bonafide/')
        os.system('mkdir -p '+feat_dir+'spoof/')
        mfcc_extractor_antispoofing(cfg, 'train')

    save_dir = cfg['ROOT_DIR'] + 'saved_models/gmm/'
    if not os.path.exists(save_dir):
        os.system('mkdir -p '+save_dir)

    bnfd_list = os.listdir(feat_dir+'bonafide/')
    spf_list = os.listdir(feat_dir+'spoof/')
    all_feats = []

    for k in range(len(bnfd_list)):
        print('Prepare training features for bonafide data {}/{} {}'.format(str(k+1), str(len(bnfd_list)), bnfd_list[k]))
        bfeat = np.load(feat_dir+'bonafide/'+bnfd_list[k])
        bfeat = (bfeat-np.mean(bfeat,axis=1,keepdims=True))/np.std(bfeat,axis=1,keepdims=True)
        all_feats.append(bfeat)

    all_feats = np.hstack(all_feats)
    bnfd_gmm = gmm(n_components=cfg['N_CPNTS'], max_iter=1, init_params='random', warm_start=True, verbose=2, verbose_interval=1)
    for k in range(10):
        print('Bonafide iteration ', k+1)
        bnfd_gmm.fit(all_feats.T)
        joblib.dump(bnfd_gmm, save_dir+'{}_bonafide_{}.gmm'.format(ctime, str(k+1)))

    all_feats = []
    for k in range(len(spf_list)):
        print('Prepare training features for spoof data {}/{} {}'.format(str(k+1), str(len(spf_list)), spf_list[k]))
        sfeat = np.load(feat_dir+'spoof/'+spf_list[k])
        sfeat = (sfeat-np.mean(sfeat,axis=1,keepdims=True))/np.std(sfeat,axis=1,keepdims=True)
        all_feats.append(sfeat)

    all_feats = np.hstack(all_feats)
    spf_gmm = gmm(n_components=cfg['N_CPNTS'], max_iter=1, init_params='random', warm_start=True, verbose=2, verbose_interval=1)
    for k in range(10):
        print('Spoof iteration ', k+1)
        spf_gmm.fit(all_feats.T)
        joblib.dump(spf_gmm, save_dir+'{}_spoof_{}.gmm'.format(ctime, str(k+1)))

def validate(cfg, ctime):
    feat_dir = cfg['DATA_DIR']+'features/mfcc_antispoofing/dev/'
    if not os.path.exists(feat_dir):
        os.system('mkdir -p '+feat_dir+'bonafide/')
        os.system('mkdir -p '+feat_dir+'spoof/')
        mfcc_extractor_antispoofing(cfg, 'dev')

    # Load models.
    #bnfd_gmm = joblib.load(cfg['ROOT_DIR']+'saved_models/gmm/{}_bonafide.gmm'.format(ctime))
    #spf_gmm = joblib.load(cfg['ROOT_DIR']+'saved_models/gmm/{}_spoof.gmm'.format(ctime))
    bnfd_gmm = joblib.load(cfg['ROOT_DIR']+'saved_models/gmm/20-01-11_06-43-31_bonafide_8.gmm')
    spf_gmm = joblib.load(cfg['ROOT_DIR']+'saved_models/gmm/20-01-11_06-43-31_spoof_8.gmm')

    bnfd_list = os.listdir(feat_dir+'bonafide/')
    spf_list = os.listdir(feat_dir+'spoof/')
    frames_per_utt = cfg['FRAMES_PER_UTT']
    bnfd_score = np.zeros(len(bnfd_list),)
    spf_score = np.zeros(len(spf_list),)

    print('Bonafide data.')
    for k, fn in enumerate(bnfd_list):
        feat = np.load(feat_dir+'bonafide/'+fn)
        #feat = (feat-np.mean(feat,axis=1,keepdims=True))/np.std(feat,axis=1,keepdims=True)
        if feat.shape[1] > frames_per_utt:
            feat = feat[:, :frames_per_utt]
        lp_b = np.mean(bnfd_gmm.score_samples(feat.T))
        lp_s = np.mean(spf_gmm.score_samples(feat.T))
        bnfd_score[k] = lp_b-lp_s
        print('{}/{} {} score:{}'.format(str(k+1), str(len(bnfd_list)), fn, str(lp_b-lp_s)))
        #print('{}/{} score:{} {}'.format(str(k+1), str(len(bnfd_list)), str(lp_b), str(lp_s)))

    print('Spoof data.')
    for k, fn in enumerate(spf_list):
        feat = np.load(feat_dir+'spoof/'+fn)
        #feat = (feat-np.mean(feat,axis=1,keepdims=True))/np.std(feat,axis=1,keepdims=True)
        if feat.shape[1] > frames_per_utt:
            feat = feat[:, :frames_per_utt]
        lp_b = np.mean(bnfd_gmm.score_samples(feat.T))
        lp_s = np.mean(spf_gmm.score_samples(feat.T))
        spf_score[k] = lp_b-lp_s
        print('{}/{} {} score:{}'.format(str(k+1), str(len(spf_list)), fn, str(lp_b-lp_s)))
        #print('{}/{} score:{} {}'.format(str(k+1), str(len(spf_list)), str(lp_b), str(lp_s)))

    thres = np.linspace(-10,10,101)
    for i in range(len(thres)):
        FRR = np.sum(bnfd_score<thres[i])/len(bnfd_list)
        FAR = np.sum(spf_score>thres[i])/len(spf_list)
        print('Threshold {}: FAR {}, FRR {}.'.format(str(thres[i]), str(FAR), str(FRR)))

def evaluate(cfg, ctime):
    feat_dir = cfg['DATA_DIR']+'features/mfcc_antispoofing/eval/'
    if not os.path.exists(feat_dir):
        os.system('mkdir -p '+feat_dir+'bonafide/')
        os.system('mkdir -p '+feat_dir+'spoof/')
        mfcc_extractor_antispoofing(cfg, 'eval')
