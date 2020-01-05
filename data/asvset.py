import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np 
import librosa
import soundfile as sf
from scipy import signal
from data.vad import ASVvad

class AttackSet(Dataset):

    def __init__(self, cfg, mode):
        self.cfg = cfg
        self.data_dir = cfg['DATA_DIR']

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass

def preprocess(cfg, mode):
    data_dir = cfg['DATA_DIR'] + 'ASVspoof2019_LA_{}/flac/'.format(mode)
    vad_data_dir = cfg['DATA_DIR'] + 'ASVspoof2019_LA_{}/vad/'.format(mode)
    if not os.path.exists(vad_data_dir):
        os.system('mkdir -p '+vad_data_dir)
    audio_list = os.listdir(data_dir)

    for i, file in enumerate(audio_list):
        print('Trimming silence of files {}/{} {}'.format(str(i+1), str(len(audio_list)), file))
        with open(data_dir+file, 'rb') as f:
            audio, sr = sf.read(f)
            assert sr == cfg['SR']

        # Convert .flac to .wav
        sf.write(data_dir+'{}.wav'.format(file[:-5]), audio, sr)

        # VAD
        fn_src = data_dir+'{}.wav'.format(file[:-5])
        fn_dst = vad_data_dir+file[:-5]
        ASVvad(fn_src, fn_dst, cfg['VAD_AGG'])

    # Remove data_dir to save disk space.
    os.system('rm -rf '+data_dir)

    # Load original data protocols.
    proto_fn = cfg['DATA_DIR'] + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.{}.trl.txt'.format(mode)
    if mode == 'train':
        proto_fn = cfg['DATA_DIR'] + 'ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt'
    with open(proto_fn, 'r') as f:
        protocols = f.readlines()
        proto_files = []
        proto_indicator = np.zeros(len(protocols),)
        for k, line in enumerate(protocols):
            info = line.strip().split()
            proto_files.append(info[1])
            proto_indicator[k] = (info[-1]=='bonafide')

    vad_audio_list = os.listdir(vad_data_dir)
    new_proto = open(cfg['DATA_DIR']+'ASVspoof2019_LA_cm_protocols/protocols_{}.txt'.format(mode), 'w')
    for i, file in enumerate(vad_audio_list):
        # Generate new data protocols.
        origin_fn = file[:-11]
        idx = proto_files.index(origin_fn)
        indicator = 'bonafide' if proto_indicator[idx] else 'spoof'
        new_proto.write('{} {}\n'.format(file[:-4], indicator))
    new_proto.close()

def mfcc_extractor_attack(cfg, mode):
    with open(cfg['DATA_DIR']+'ASVspoof2019_LA_cm_protocols/protocols_{}.txt'.format(mode), 'r') as f:
        protocols = f.readlines()

    # Minumum of file length.   
    min_samples = cfg['FIXED_SEC']*cfg['SR']
    vad_data_dir = cfg['DATA_DIR'] + 'ASVspoof2019_LA_{}/vad/'.format(mode)

    # Extract features (spoof data) for adversarial attack.
    for i, file in enumerate(protocols):
        info = file.strip().split()
        print('For attack, extracting features from file {}/{} {}'.format(str(i+1), str(len(protocols)), info[0]+'.wav'))
        if info[1] == 'spoof':
            audio, sr = librosa.load(path=vad_data_dir+info[0]+'.wav', sr=None, mono=True)
            if len(audio) < min_samples:
                print('This audio file is too short. Do not use it.')
            else:
                # Pre-emphasis
                audio = audio[:min_samples]
                audio = np.append(audio[0], audio[1:]-cfg['PREEMPH']*audio[:-1])

                # STFT: magnitude(power) and phase
                D = librosa.core.stft(y=audio, n_fft=cfg['FFT_LEN'], hop_length=cfg['HOP_LEN'], window='hamming')
                mag = np.abs(D)**2
                phs = np.angle(D)
                
                # Mel-frequency spectrogram
                mel_fb = librosa.filters.mel(sr=sr, n_fft=cfg['FFT_LEN'], n_mels=cfg['MEL_DIM'])
                mel_spec = np.dot(mel_fb, mag)

                # MFCC
                mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=cfg['MFCC_DIM'])

                # Save: MFCC, phase
                # Save acoustic feature
                print('Save MFCC and phase spectral.')
                np.save(cfg['DATA_DIR']+'features/mfcc_attack/{}/{}.npy'.format(mode,info[0]), mfcc)
                np.save(cfg['DATA_DIR']+'features/phase_attack/{}/{}.npy'.format(mode,info[0]), phs)
            
        else:
            print('Genuine speech. Do not use it here.')

def mfcc_extractor_antispoofing(cfg, mode):
    with open(cfg['DATA_DIR']+'ASVspoof2019_LA_cm_protocols/protocols_{}.txt'.format(mode), 'r') as f:
        protocols = f.readlines()

    vad_data_dir = cfg['DATA_DIR'] + 'ASVspoof2019_LA_{}/vad/'.format(mode)

    # Extract features (spoof data) for adversarial attack.
    for i, file in enumerate(protocols):
        info = file.strip().split()
        print('For anti-spoofing, extracting features from file {}/{} {}'.format(str(i+1), str(len(protocols)), info[0]+'.wav'))
        audio, sr = librosa.load(path=vad_data_dir+info[0]+'.wav', sr=None, mono=True)

        # Pre-emphasis
        audio = np.append(audio[0], audio[1:]-cfg['PREEMPH']*audio[:-1])

        # STFT: magnitude(power) and phase
        D = librosa.core.stft(y=audio, n_fft=cfg['FFT_LEN'], hop_length=cfg['HOP_LEN'], window='hamming')
        mag = np.abs(D)**2
        
        # Mel-frequency spectrogram
        mel_fb = librosa.filters.mel(sr=sr, n_fft=cfg['FFT_LEN'], n_mels=cfg['MEL_DIM'])
        mel_spec = np.dot(mel_fb, mag)

        # MFCC
        mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel_spec), n_mfcc=cfg['MFCC_DIM'])
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
        mfcc = np.vstack([mfcc, mfcc_delta, mfcc_delta2])

        # Save: MFCC
        print('Save MFCC.')
        np.save(cfg['DATA_DIR']+'features/mfcc_antispoofing/{}/{}/{}.npy'.format(mode,info[1],info[0]), mfcc)            

def mfcc_to_audio(cfg, name):

    # Load
    # mfcc_max = np.load()
    # mfcc = np.load()
    # phs = np.load()

    # MFCC to Mel-spectrogram
    mel_spec = librosa.feature.inverse.mfcc_to_mel(mfcc=mfcc, n_mels=cfg['MEL_DIM'])

    # Mel-spectrogram to magnitude(power)
    mag = librosa.feature.inverse.mel_to_stft(M=mel_spec, sr=cfg['SR'], n_fft=cfg['FFT_LEN'])

    # ISTFT
    j = 1j
    D = np.sqrt(mag)*np.cos(phs) + j*np.sqrt(mag)*np.sin(phs)
    audio = librosa.core.istft(stft_matrix=D, hop_length=cfg['HOP_LEN'], window='hamming')
    audio = signal.lfilter([1], [1, -cfg['PREEMPH']], audio)

    return audio