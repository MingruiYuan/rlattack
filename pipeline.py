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

# LFCC_THRESHOLD = 0.748917
# CQCC_THRESHOLD = 1.252953

def remove_inf(audio):  
    inf_pos = np.where(np.isfinite(audio)==False)[0]
    for p in inf_pos:
        if p == 0:
            audio[p] = audio[1]
        elif p == len(audio)-1:
            audio[p] = audio[-2]
        else:
            audio[p] = (audio[p+1] + audio[p-1])/2
    return audio

def get_score_resnet(cfg, audio, asvmodel, mode='train'):
    audio = librosa.util.normalize(pad(audio, cfg['MAX_PADLEN']))
    mfcc = librosa.feature.mfcc(audio, sr=cfg['SR'], n_mfcc=cfg['MFCC_DIM'])
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(delta)
    mfcc = np.expand_dims(np.concatenate((mfcc, delta, delta2), axis=0), axis=0)
    mfcc = torch.from_numpy(mfcc).to(device)
    with torch.no_grad():
        output = asvmodel(mfcc)
    score = output[0][1].item() - output[0][0].item()
    if mode == 'eval':
        print('Score', score)
        reward = score > cfg['DL_THRESHOLD']
        return reward
    else:
        return score

def get_score_gmm(cfg, ctime, audio, meng, mode='train'):
    filename = cfg['ROOT_DIR'] + 'temp_audio/temp_audio_{}.wav'.format(ctime)
    sf.write(filename, audio, cfg['SR'])
    score = meng.get_score(cfg['TOOLKIT_DIR'], cfg['FT_TYPE'], filename)
    if mode == 'eval':
        print('Score', score)
        reward = score > cfg['THRESHOLD']
        return reward
    else:
        return score
        
def train(cfg, ctime, oracle='GMM', LO=False, load_actor_path=None, load_critic_path=None, load_asvmodel_path=None):
    print(torch.cuda.is_available())
    feat_dir = cfg['DATA_DIR']+'features/logmel_attack/train/'
    phase_dir = cfg['DATA_DIR']+'features/phase_attack/train/'
    val_feat_dir = cfg['DATA_DIR']+'features/logmel_attack/dev/'
    val_phase_dir = cfg['DATA_DIR']+'features/phase_attack/dev/'
    mfcc_extractor_attack(cfg, 'train')
    feat_list = os.listdir(feat_dir)
    random.shuffle(feat_list)
    mfcc_extractor_attack(cfg, 'dev')
    val_feat_list = os.listdir(val_feat_dir)

    if not os.path.exists(cfg['ROOT_DIR']+'evading_audio/{}/'.format(ctime)):
        os.system('mkdir -p '+cfg['ROOT_DIR']+'evading_audio/{}/'.format(ctime))
    if not os.path.exists(cfg['ROOT_DIR']+'figures/'):
        os.system('mkdir -p '+cfg['ROOT_DIR']+'figures/')
    if not os.path.exists(cfg['ROOT_DIR']+'temp_audio/'):
        os.system('mkdir -p '+cfg['ROOT_DIR']+'temp_audio/')

    agent = DDPG(cfg)
    agent.load_model(load_actor_path, load_critic_path)
    action_noise = ActionNoise(cfg['MEL_DIM'], cfg['FRAMES_PER_UTT']) if cfg['ACTION_NOISE'] else None
    memory = ReplayMemory(cfg['MEM_SIZE'])
    rewards = []
    val_rewards = []
    per_action_rewards = []
    val_per_action_rewards = []
    episode = 0
    count_file = 0
    update = 0
    interval_success = 0
    
    if oracle == 'GMM':
        meng = matlab.engine.start_matlab()
    if oracle == 'ResNet':
        asvmodel = MFCCModel()
        asvmodel_ckpt = torch.load(load_asvmodel_path, map_location=device)
        asvmodel.load_state_dict(asvmodel_ckpt['model_state_dict'])
        asvmodel.to(device)
        asvmodel.eval()

    while episode < cfg['MAX_EPISODE']:
        filename = feat_list[count_file % len(feat_list)][:-4]
        state = np.expand_dims(np.load(feat_dir+feat_list[count_file % len(feat_list)]), axis=0)
        state = torch.from_numpy(state).to(device)
        phase = np.load(phase_dir+feat_list[count_file % len(feat_list)])
        episode_reward = 0
        iteration = 0

        if action_noise is not None:
            action_noise.scale = (cfg['INIT_EXPSCALE'] - cfg['FINAL_EXPSCALE'])*max(0, cfg['EXPLORATION_END'] - episode)/cfg['EXPLORATION_END'] + cfg['FINAL_EXPSCALE']
            action_noise.reset()

        print('Episode', episode+1, filename)
        if torch.cuda.is_available():    
            base_audio = mfcc_to_audio(cfg, state.squeeze().cpu().numpy(), phase)
        else:
            base_audio = mfcc_to_audio(cfg, state.squeeze().numpy(), phase)
        if not np.all(np.isfinite(base_audio)):
            print('Infinite value. Drop this one.')
            count_file += 1
            feat_list.remove(filename+'.npy')
            continue

        if oracle == 'GMM':
            base_score = get_score_gmm(cfg, ctime, base_audio, meng)
        if oracle == 'ResNet':
            base_score = get_score_resnet(cfg, base_audio, asvmodel)
        if base_score > cfg['THRESHOLD' if oracle == 'GMM' else 'DL_THRESHOLD']:
            print('False accept case. Skip this one.')
            count_file += 1
            feat_list.remove(filename+'.npy')
            continue

        while iteration < cfg['ITER_PER_UTT']:
            action = agent.select_action(state, action_noise)
            next_state = torch.add(state.squeeze(), action)

            ## Evaluate perturbed features ##
            if torch.cuda.is_available():    
                audio = mfcc_to_audio(cfg, next_state.cpu().numpy(), phase)
            else:
                audio = mfcc_to_audio(cfg, next_state.numpy(), phase)
            if not np.all(np.isfinite(audio)):
                print('Infinite value. Drop this one.')
                break

            if oracle == 'GMM':
                if LO:
                    reward = get_score_gmm(cfg, ctime, audio, meng, 'eval')
                else:
                    score = get_score_gmm(cfg, ctime, audio, meng)
                    reward = score - base_score
                    base_score = score              
            if oracle == 'ResNet':
                if LO:
                    reward = get_score_resnet(cfg, audio, asvmodel, 'eval')
                else:
                    score = get_score_resnet(cfg, audio, asvmodel)
                    reward = score - base_score
                    base_score = score
            
            episode_reward += reward
            if LO:
                mask = torch.Tensor([not reward])
                print(iteration+1, 'Reward', reward)
            else:
                mask = torch.Tensor([not (score>cfg['THRESHOLD' if oracle == 'GMM' else 'DL_THRESHOLD'])])           
                print(iteration+1, 'Score', score, 'Reward', reward)        

            if torch.cuda.is_available():
                memory.push(state.squeeze().cpu(), action.cpu(), mask, next_state.cpu(), torch.Tensor([reward]))
            else:
                memory.push(state.squeeze(), action, mask, next_state, torch.Tensor([reward]))

            success_flag = reward if LO else score > cfg['THRESHOLD' if oracle == 'GMM' else 'DL_THRESHOLD']
            if success_flag:
                interval_success += 1                
                evading_audio_path = cfg['ROOT_DIR']+'evading_audio/{}/{}_{}.wav'.format(ctime, filename, str(episode//len(feat_list)))
                sf.write(evading_audio_path, audio, cfg['SR'])
                print('Succeed!!')
                break

            state = torch.unsqueeze(next_state, dim=0)
            iteration += 1
            
            if len(memory) > cfg['ATK_BS']:
                transitions = memory.sample(cfg['ATK_BS'])
                batch = Transition(*zip(*transitions))
                value_loss, policy_loss = agent.update_parameters(batch)
                if update % cfg['SAVE_INTERVAL'] == 0:
                    agent.save_model(ctime, update)
                update += 1
                print('Update {}: Value loss {}, Policy loss {}'.format(str(update), str(value_loss), str(policy_loss)))
        
        rewards.append(episode_reward)
        per_action_rewards.append(episode_reward/(iteration+1))
        print('Episode reward {}, with {} iterations.'.format(str(episode_reward), str(iteration+1) if iteration<cfg['ITER_PER_UTT'] else 'N/A'))
        print('Reward per action {}.'.format(episode_reward/(iteration+1)))
        fig, ax = plt.subplots()
        ax.plot(per_action_rewards)
        fig.savefig(cfg['ROOT_DIR']+'figures/per_action_rewards_{}.png'.format(ctime))
        plt.close(fig)
        np.save(cfg['ROOT_DIR']+'figures/episode_rewards_{}.npy'.format(ctime), np.array(rewards))
        np.save(cfg['ROOT_DIR']+'figures/per_action_rewards_{}.npy'.format(ctime), np.array(per_action_rewards))
        if episode > 19:
            print('Average episode reward: ', np.mean(rewards[-20:]))     

        if episode % cfg['VAL_INTERVAL'] == 0:
            print('VALIDATION at episode ', episode+1)
            print('Interval success rate: {}/{}.'.format(str(interval_success), cfg['VAL_INTERVAL']))
            interval_success = 0           
            random.shuffle(val_feat_list)
            val_number = 0
            val_count_file = 0
            average_val_reward = 0
            val_success = 0

            while val_number < cfg['VAL_BS']:
                val_filename = val_feat_list[val_count_file][:-4]
                state = np.expand_dims(np.load(val_feat_dir+val_feat_list[val_count_file]), axis=0)
                state = torch.from_numpy(state).to(device)
                phase = np.load(val_phase_dir+val_feat_list[val_count_file])
                episode_reward = 0
                iteration = 0

                print('Validation Number', val_number+1, val_filename)
                if torch.cuda.is_available():    
                    base_audio = mfcc_to_audio(cfg, state.squeeze().cpu().numpy(), phase)
                else:
                    base_audio = mfcc_to_audio(cfg, state.squeeze().numpy(), phase)
                if not np.all(np.isfinite(base_audio)):
                    print('Infinite value. Drop this one.')
                    val_count_file += 1
                    val_feat_list.remove(val_filename+'.npy')
                    continue

                if oracle == 'GMM':
                    base_score = get_score_gmm(cfg, ctime, base_audio, meng)
                if oracle == 'ResNet':
                    base_score = get_score_resnet(cfg, base_audio, asvmodel)
                if base_score > cfg['THRESHOLD' if oracle == 'GMM' else 'DL_THRESHOLD']:
                    print('False accept case. Skip this one.')
                    val_count_file += 1
                    val_feat_list.remove(val_filename+'.npy')
                    continue

                while iteration < cfg['ITER_PER_UTT']:
                    action = agent.select_action(state)
                    next_state = torch.add(state.squeeze(), action)

                    if torch.cuda.is_available():
                        audio = mfcc_to_audio(cfg, next_state.cpu().numpy(), phase)
                    else:
                        audio = mfcc_to_audio(cfg, next_state.numpy(), phase)
                    if not np.all(np.isfinite(audio)):
                        print('Infinite value. Drop this one.')
                        break
                    
                    if oracle == 'GMM':
                        score = get_score_gmm(cfg, ctime, audio, meng)
                    if oracle == 'ResNet':
                        score = get_score_resnet(cfg, audio, asvmodel)
                    reward = score - base_score
                    base_score = score
                    
                    episode_reward += reward
                    print(iteration+1, 'Score', score, 'Reward', reward)

                    if score > cfg['THRESHOLD' if oracle == 'GMM' else 'DL_THRESHOLD']:
                        val_success += 1
                        evading_audio_path = cfg['ROOT_DIR']+'evading_audio/{}/{}_{}.wav'.format(ctime, val_filename, str(episode))
                        sf.write(evading_audio_path, audio, cfg['SR'])
                        break

                    state = torch.unsqueeze(next_state, dim=0)
                    iteration += 1

                val_number += 1
                val_count_file += 1
                average_val_reward += episode_reward
                val_rewards.append(episode_reward)
                val_per_action_rewards.append(episode_reward/(iteration+1))
                print('{}/{}. Episode reward: {}, with {} iterations.'.format(str(val_number), str(cfg['VAL_BS']), str(episode_reward), str(iteration+1) if iteration<cfg['ITER_PER_UTT'] else 'N/A'))
                print('Reward per action {}.'.format(episode_reward/(iteration+1)))
                fig, ax = plt.subplots()
                ax.plot(val_per_action_rewards)
                fig.savefig(cfg['ROOT_DIR']+'figures/val_per_action_rewards_{}.png'.format(ctime))
                plt.close(fig)
                np.save(cfg['ROOT_DIR']+'figures/val_episode_rewards_{}.npy'.format(ctime), np.array(val_rewards))
                np.save(cfg['ROOT_DIR']+'figures/val_per_action_rewards_{}.npy'.format(ctime), np.array(val_per_action_rewards))

            print('Average validation reward at episode {}: {}.'.format(str(episode+1), str(average_val_reward/cfg['VAL_BS'])))
            print('Validation success rate: {}/{}.'.format(str(val_success), cfg['VAL_BS']))
                    
        episode += 1
        count_file += 1
    
    if oracle == 'GMM':
        meng.quit()

def evaluate(cfg, ctime, oracle='GMM', load_actor_path=None, load_critic_path=None, load_asvmodel_path=None):
    print(torch.cuda.is_available())
    feat_dir = cfg['DATA_DIR']+'features/logmel_attack/eval/'
    phase_dir = cfg['DATA_DIR']+'features/phase_attack/eval/'
    mfcc_extractor_attack(cfg, 'eval')
    feat_list = os.listdir(feat_dir)
    random.shuffle(feat_list)

    if not os.path.exists(cfg['ROOT_DIR']+'evading_audio/{}/'.format(ctime)):
        os.system('mkdir -p '+cfg['ROOT_DIR']+'evading_audio/{}/'.format(ctime))
    if not os.path.exists(cfg['ROOT_DIR']+'temp_audio/'):
        os.system('mkdir -p '+cfg['ROOT_DIR']+'temp_audio/')

    agent = DDPG(cfg)
    agent.load_model(load_actor_path, load_critic_path)
    rewards = []
    count = 0
    count_file = 0
    
    if oracle == 'GMM':
        meng = matlab.engine.start_matlab()
    if oracle == 'ResNet':
        asvmodel = MFCCModel()
        asvmodel_ckpt = torch.load(load_asvmodel_path, map_location=device)
        asvmodel.load_state_dict(asvmodel_ckpt['model_state_dict'])
        asvmodel.to(device)
        asvmodel.eval()

    while count < cfg['EVAL_NUM']:
        filename = feat_list[count_file][:-4]
        state = np.expand_dims(np.load(feat_dir+feat_list[count_file]), axis=0)
        state = torch.from_numpy(state).to(device)
        phase = np.load(phase_dir+feat_list[count_file])
        episode_reward = 0
        iteration = 0

        print('Evaluation', count+1, filename)
        if torch.cuda.is_available():    
            base_audio = mfcc_to_audio(cfg, state.squeeze().cpu().numpy(), phase)
        else:
            base_audio = mfcc_to_audio(cfg, state.squeeze().numpy(), phase)
        if not np.all(np.isfinite(base_audio)):
            print('Infinite value. Drop this one.')
            count_file += 1
            feat_list.remove(filename+'.npy')
            continue

        if oracle == 'GMM': 
            base_reward = get_score_gmm(cfg, ctime, base_audio, meng, 'eval')
        if oracle == 'ResNet':
            base_reward = get_score_resnet(cfg, base_audio, asvmodel, 'eval')
        if base_reward:
            print('False accept case. Skip this one.')
            count_file += 1
            feat_list.remove(filename+'.npy')
            continue

        while iteration < cfg['EVAL_ITER_PER_UTT']:
            print(iteration+1, '/', cfg['EVAL_ITER_PER_UTT'])
            action = agent.select_action(state)
            next_state = torch.add(state.squeeze(), action)

            ## Evaluate perturbed features ##
            if torch.cuda.is_available():    
                audio = mfcc_to_audio(cfg, next_state.cpu().numpy(), phase)
            else:
                audio = mfcc_to_audio(cfg, next_state.numpy(), phase)
            if not np.all(np.isfinite(audio)):
                print('Infinite value. Drop this one.')
                break

            if oracle == 'GMM':
                reward = get_score_gmm(cfg, ctime, audio, meng, 'eval')
            if oracle == 'ResNet':
                reward = get_score_resnet(cfg, audio, asvmodel, 'eval')            
            episode_reward += reward

            if reward:
                evading_audio_path = cfg['ROOT_DIR']+'evading_audio/{}/{}_{}.wav'.format(ctime, filename, 'ptb')
                sf.write(evading_audio_path, audio, cfg['SR'])
                print('Succeed!!')
                break

            state = torch.unsqueeze(next_state, dim=0)
            iteration += 1

        rewards.append(episode_reward)
        print('Episode reward: {}, with {} iterations.'.format(str(episode_reward), str(iteration+1) if episode_reward else 'N/A'))
        print('Current successful rate: {}/{}'.format(sum(rewards), len(rewards)))

        count += 1
        count_file += 1
    
    if oracle == 'GMM':
        meng.quit()

    print('Successful rate: {}/{}.'.format(sum(rewards), cfg['EVAL_NUM']))