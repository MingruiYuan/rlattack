import os, json
import argparse

if __name__ == '__main__':
    ps = argparse.ArgumentParser(description='Attack Anti-spoofing Systems')
    ps.add_argument('-C', '--config', type=str, default=None)
    ps.add_argument('-T', '--current_time', type=str, required=True)
    args = ps.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)
    
    from mfcc_gmm.pipeline import train as mfcc_gmm_train
    mfcc_gmm_train(cfg, args.current_time)