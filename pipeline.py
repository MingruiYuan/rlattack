import matlab.engine
import os, random
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from data.asvset import mfcc_extractor_attack, mfcc_to_audio, pad
from antispoof_resnet.models import MFCCModel
from engine import DDPG, ActionNoise
from replay_memory import ReplayMemory, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_reward_resnet(cfg, audio, asvmodel):
    audio = librosa.util.normalize(pad(audio, cfg['MAX_PADLEN']))
    mfcc = librosa.feature.mfcc(audio, sr=cfg['SR'], n_mfcc=cfg['MFCC_DIM'])
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(delta)
    mfcc = np.expand_dims(np.concatenate((mfcc, delta, delta2), axis=0), axis=0)
    mfcc = torch.from_numpy(mfcc).to(device)
    _, reward = asvmodel(mfcc).max(dim=1)
    return reward.item()

def get_reward_lfcc(cfg, ctime, audio, meng, mode='train'):
    if not os.path.exists(cfg['AUDIO_SAVE_DIR']):
        os.system('mkdir -p '+ cfg['AUDIO_SAVE_DIR'])
    filename = cfg['AUDIO_SAVE_DIR'] + 'temp_audio_{}.wav'.format(ctime)
    sf.write(filename, audio, cfg['SR'])
    reward = meng.get_reward(cfg['TOOLKIT_DIR'], filename)
    if mode == 'eval':
        print('Reward ', reward)
        reward = reward > cfg['THRESHOLD']
        
    return reward

def train(cfg, ctime, load_actor_path=None, load_critic_path=None):
    print(torch.cuda.is_available())
    mfcc_dir = cfg['DATA_DIR']+'features/logmel_attack/train/'
    phase_dir = cfg['DATA_DIR']+'features/phase_attack/train/'
    val_mfcc_dir = cfg['DATA_DIR']+'features/logmel_attack/dev/'
    val_phase_dir = cfg['DATA_DIR']+'features/phase_attack/dev/'
    mfcc_extractor_attack(cfg, 'train')
    mfcc_list = os.listdir(mfcc_dir)
    mfcc_extractor_attack(cfg, 'dev')
    val_mfcc_list = os.listdir(val_mfcc_dir)

    ## anti-spoofing model ##
    # asvmodel = MFCCModel()
    # asvmodel_ckpt = torch.load(cfg['ROOT_DIR']+'antispoofing.pth')
    # asvmodel.load_state_dict(asvmodel_ckpt['model_state_dict'])
    # asvmodel.to(device)
    # asvmodel.eval()

    agent = DDPG(cfg)
    action_noise = ActionNoise(cfg['MFCC_DIM'], cfg['FRAMES_PER_UTT']) if cfg['ACTION_NOISE'] else None
    memory = ReplayMemory(cfg['MEM_SIZE'])
    rewards = []
    episode = 0
    update = 0

    agent.load_model(load_actor_path, load_critic_path)
    meng = matlab.engine.start_matlab()

    while episode < cfg['MAX_EPISODE']:
        filename = mfcc_list[episode % len(mfcc_list)][:-4]
        state = np.expand_dims(np.load(mfcc_dir+mfcc_list[episode % len(mfcc_list)]), axis=0)
        state = torch.from_numpy(state).to(device)
        phase = np.load(phase_dir+mfcc_list[episode % len(mfcc_list)])
        episode_reward = 0
        iteration = 0

        if action_noise is not None:
            action_noise.scale = (cfg['INIT_EXPSCALE'] - cfg['FINAL_EXPSCALE'])*max(0, cfg['EXPLORATION_END'] - episode)/cfg['EXPLORATION_END'] + cfg['FINAL_EXPSCALE']
            action_noise.reset()

        print('Episode ', episode+1, filename)

        while iteration < cfg['ITER_PER_UTT']:
            action = agent.select_action(state, action_noise)
            next_state = torch.add(state.squeeze(), action)

            ## Evaluate perturbed features ##
            if torch.cuda.is_available():    
                audio = mfcc_to_audio(cfg, next_state.cpu().numpy(), phase)
            else:
                audio = mfcc_to_audio(cfg, next_state.numpy(), phase)

            # reward = get_reward_resnet(cfg, audio, asvmodel)
            reward = get_reward_lfcc(cfg, ctime, audio, meng)
            
            # sf.write(cfg['AUDIO_SAVE_DIR'], audio, cfg['SR'])
            # reward = meng.get_reward(cfg['TOOLKIT_DIR'], cfg['AUDIO_SAVE_DIR'])            
            episode_reward += reward
            mask = torch.Tensor([not (reward>cfg['THRESHOLD'])])
            print('Reward ', iteration+1, reward)

            memory.push(state.squeeze(), action, mask, next_state, torch.Tensor([reward]))

            if reward > cfg['THRESHOLD']:
                if not os.path.exists(cfg['ROOT_DIR']+'evading_audio/{}/'.format(ctime)):
                    os.system('mkdir -p '+cfg['ROOT_DIR']+'evading_audio/{}/'.format(ctime))
                evading_audio_path = cfg['ROOT_DIR']+'evading_audio/{}/{}_{}.wav'.format(ctime, filename, str(episode//len(mfcc_list)))
                sf.write(evading_audio_path, audio, cfg['SR'])
                print('Succeed!!')
                break

            state = torch.unsqueeze(next_state, dim=0)
            iteration += 1
            
            if len(memory) > cfg['ATK_BS']:
                transitions = memory.sample(cfg['ATK_BS'])
                batch = Transition(*zip(*transitions))
                value_loss, policy_loss = agent.update_parameters(batch)
                if update % 1000 == 0:
                    agent.save_model(ctime, update)
                update += 1
                print('Update {}: Value loss {}, Policy loss {}'.format(str(update), str(value_loss), str(policy_loss)))
        
        rewards.append(episode_reward)
        print('Episode reward: {}, with {} iterations.'.format(str(episode_reward), str(iteration+1) if iteration<cfg['ITER_PER_UTT'] else 'N/A'))
        if episode > 19:
            print('Average episode reward: ', np.mean(rewards[-20:]))     

        if episode % 100 == 0:
            print('VALIDATION at episode ', episode+1)           
            random.shuffle(val_mfcc_list)
            val_number = 0

            while val_number < cfg['VAL_BS']:
                val_filename = val_mfcc_list[val_number][:-4]
                state = np.expand_dims(np.load(val_mfcc_dir+val_mfcc_list[val_number]), axis=0)
                state = torch.from_numpy(state).to(device)
                phase = np.load(val_phase_dir+val_mfcc_list[val_number])
                episode_reward = 0
                average_val_reward = 0
                iteration = 0
                print('Validation Number ', val_number+1, val_filename)

                while iteration < cfg['ITER_PER_UTT']:
                    action = agent.select_action(state)
                    next_state = torch.add(state.squeeze(), action)

                    if torch.cuda.is_available():
                        audio = mfcc_to_audio(cfg, next_state.cpu().numpy(), phase)
                    else:
                        audio = mfcc_to_audio(cfg, next_state.numpy(), phase)
                    
                    # reward = get_reward_resnet(cfg, audio, asvmodel)
                    reward = get_reward_lfcc(cfg, ctime, audio, meng)
                    print('Reward ', iteration+1, reward)
                    # sf.write(cfg['AUDIO_SAVE_DIR'], audio, cfg['SR'])
                    # reward = meng.get_reward(cfg['TOOLKIT_DIR'], cfg['AUDIO_SAVE_DIR'])
                    episode_reward += reward

                    if reward > cfg['THRESHOLD']:
                        if not os.path.exists(cfg['ROOT_DIR']+'evading_audio/{}/'.format(ctime)):
                            os.system('mkdir -p '+cfg['ROOT_DIR']+'evading_audio/{}/'.format(ctime))
                        evading_audio_path = cfg['ROOT_DIR']+'evading_audio/{}/{}_{}.wav'.format(ctime, val_filename, str(episode))
                        sf.write(evading_audio_path, audio, cfg['SR'])
                        break

                    state = torch.unsqueeze(next_state, dim=0)
                    iteration += 1

                val_number += 1
                average_val_reward += episode_reward
                print('{}/{}. Episode reward: {}, with {} iterations.'.format(str(val_number), str(cfg['VAL_BS']), str(episode_reward), str(iteration+1) if iteration<cfg['ITER_PER_UTT'] else 'N/A'))

            print('Average validation reward at episode {}: {}.'.format(str(episode+1), str(average_val_reward/cfg['VAL_BS'])))
                    
        episode += 1

    meng.quit()

def evaluate(cfg, ctime, load_actor_path=None, load_critic_path=None):
    print(torch.cuda.is_available())
    feat_dir = cfg['DATA_DIR']+'features/logmel_attack/eval/'
    phase_dir = cfg['DATA_DIR']+'features/phase_attack/eval/'
    mfcc_extractor_attack(cfg, 'eval')
    feat_list = os.listdir(feat_dir)
    random.shuffle(feat_list)

    agent = DDPG(cfg)
    rewards = []
    count = 0

    agent.load_model(load_actor_path, load_critic_path)
    meng = matlab.engine.start_matlab()

    while count < cfg['EVAL_NUM']:
        filename = feat_list[count][:-4]
        state = np.expand_dims(np.load(feat_dir+feat_list[count]), axis=0)
        state = torch.from_numpy(state).to(device)
        phase = np.load(phase_dir+feat_list[count])
        episode_reward = 0
        iteration = 0

        print('Evaluation ', count+1, filename)

        while iteration < cfg['EVAL_ITER_PER_UTT']:
            print(iteration+1, '/', cfg['EVAL_ITER_PER_UTT'])
            action = agent.select_action(state)
            next_state = torch.add(state.squeeze(), action)

            ## Evaluate perturbed features ##
            if torch.cuda.is_available():    
                audio = mfcc_to_audio(cfg, next_state.cpu().numpy(), phase)
            else:
                audio = mfcc_to_audio(cfg, next_state.numpy(), phase)

            reward = get_reward_lfcc(cfg, ctime, audio, meng, 'eval')            
            episode_reward += reward

            if reward:
                if not os.path.exists(cfg['ROOT_DIR']+'evading_audio/{}/'.format(ctime)):
                    os.system('mkdir -p '+cfg['ROOT_DIR']+'evading_audio/{}/'.format(ctime))
                evading_audio_path = cfg['ROOT_DIR']+'evading_audio/{}/{}_{}.wav'.format(ctime, filename, 'ptb')
                sf.write(evading_audio_path, audio, cfg['SR'])
                print('Succeed!!')
                break

            state = torch.unsqueeze(next_state, dim=0)
            iteration += 1

        rewards.append(episode_reward)
        print('Episode reward: {}, with {} iterations.'.format(str(episode_reward), str(iteration+1) if episode_reward else 'N/A'))

        count += 1

    meng.quit()
    print('Successful rate: {}/{}.'.format(sum(rewards), cfg['EVAL_NUM']))