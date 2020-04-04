import os, json
import argparse

if __name__ == '__main__':
    ps = argparse.ArgumentParser(description='Attack Anti-spoofing Systems.')
    ps.add_argument('-S', '--step', type=str, required=True)
    ps.add_argument('-C', '--config', type=str, required=True)
    ps.add_argument('-T', '--current_time', type=str, default=None)
    ps.add_argument('--feature_type', type=str, default=None)
    ps.add_argument('--label_only', action='store_true')
    ps.add_argument('--actor_path', type=str, default=None)
    ps.add_argument('--critic_path', type=str, default=None)
    args = ps.parse_args()

    with open(args.config, 'r') as f:
        cfg = json.load(f)

    if args.step == 'preprocess':
        from data.asvset import preprocess
        preprocess(cfg, 'train')
        preprocess(cfg, 'dev')
        preprocess(cfg, 'eval')

    if args.step == 'train_attack':
        from pipeline import train as attack_train
        attack_train(cfg, args.current_time, args.feature_type, args.label_only, args.actor_path, args.critic_path)

    if args.step == 'eval_attack':
        from pipeline import evaluate as attack_eval
        attack_eval(cfg, args.current_time, args.feature_type, args.actor_path, args.critic_path)