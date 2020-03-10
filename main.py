import os, json
import argparse

if __name__ == '__main__':
    ps = argparse.ArgumentParser(description='Attack Anti-spoofing Systems.')
    ps.add_argument('-S', '--step', type=str, required=True)
    ps.add_argument('-C', '--config', type=str, default=None)
    ps.add_argument('-T', '--current_time', type=str, required=True)
    ps.add_argument('-O', '--oracle', type=str, default='GMM')
    ps.add_argument('--label_only', action='store_true')
    ps.add_argument('--actor_path', type=str, default=None)
    ps.add_argument('--critic_path', type=str, default=None)
    ps.add_argument('--asvmodel_path', type=str, default=None)
    args = ps.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)
    
    if args.step == 'train_gmm':
        from mfcc_gmm.pipeline import train_warmstart as mfcc_gmm_train
        mfcc_gmm_train(cfg, args.current_time)

    if args.step == 'val_gmm':
        from mfcc_gmm.pipeline import validate as mfcc_gmm_val
        mfcc_gmm_val(cfg, args.current_time)

    if args.step == 'train_asvresnet':
        from antispoof_resnet.pipeline import train as asvresnet_train
        asvresnet_train(cfg, args.current_time, cfg['LOAD_ASVRESNET_PATH'])

    if args.step == 'eval_asvresnet':
        from antispoof_resnet.pipeline import evaluate as asvresnet_eval
        asvresnet_eval(cfg, cfg['LOAD_ASVRESNET_PATH'])

    if args.step == 'train_attack':
        from pipeline import train as attack_train
        attack_train(cfg, args.current_time, args.oracle, args.label_only, args.actor_path, args.critic_path, args.asvmodel_path)

    if args.step == 'eval_attack':
        from pipeline import evaluate as attack_eval
        attack_eval(cfg, args.current_time, args.oracle, args.actor_path, args.critic_path, args.asvmodel_path)