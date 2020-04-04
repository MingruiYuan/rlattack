import matlab.engine
import os, random
import torch
import numpy as np
import librosa
import soundfile as sf
import matplotlib.pyplot as plt
from data.asvset import feature_extractor_attack, feature_to_audio, pad
from engine import DDPG, ActionNoise
from replay_memory import ReplayMemory, Transition

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_score_gmm(cfg, ctime, audio, meng, ft, mode='train', thres=0):
    filename = cfg['ROOT_DIR'] + 'temp_audio/temp_audio_{}.wav'.format(ctime)
    sf.write(filename, audio, cfg['SR'])
    score = meng.get_score(cfg['TOOLKIT_DIR'], ft, filename)
    if mode == 'eval':
        print('Score', score)
        reward = score > thres
        return reward
    else:
        return score
        
def train(cfg, ctime, feature_type, LO=False, load_actor_path=None, load_critic_path=None):
    print(torch.cuda.is_available())
    feat_dir = cfg['DATA_DIR']+'features/logmel_attack/train/'
    phase_dir = cfg['DATA_DIR']+'features/phase_attack/train/'
    val_feat_dir = cfg['DATA_DIR']+'features/logmel_attack/dev/'
    val_phase_dir = cfg['DATA_DIR']+'features/phase_attack/dev/'
    feature_extractor_attack(cfg, 'train')
    feat_list = os.listdir(feat_dir)
    random.shuffle(feat_list)
    feature_extractor_attack(cfg, 'dev')
    val_feat_list = os.listdir(val_feat_dir)

    if not os.path.exists(cfg['ROOT_DIR']+'evading_audio/{}/'.format(ctime)):
        os.system('mkdir -p '+cfg['ROOT_DIR']+'evading_audio/{}/'.format(ctime))  
    if not os.path.exists(cfg['ROOT_DIR']+'temp_audio/'):
        os.system('mkdir -p '+cfg['ROOT_DIR']+'temp_audio/')

    agent = DDPG(cfg, feature_type)
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
    
    meng = matlab.engine.start_matlab()
    if feature_type == 'LFCC':
        threshold = 0.748917
    elif feature_type == 'CQCC':
        threshold = 1.252953
    else:
        print('Feature type not available.')  

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
            base_audio = feature_to_audio(cfg, state.squeeze().cpu().numpy(), phase)
        else:
            base_audio = feature_to_audio(cfg, state.squeeze().numpy(), phase)

        base_score = get_score_gmm(cfg, ctime, base_audio, meng, feature_type)
        if base_score > threshold:
            print('False accept case. Skip this one.')
            count_file += 1
            feat_list.remove(filename+'.npy')
            continue

        while iteration < cfg['ITER_PER_UTT']:
            action = agent.select_action(state, action_noise)
            next_state = torch.add(state.squeeze(), action)

            ## Evaluate perturbed features ##
            if torch.cuda.is_available():    
                audio = feature_to_audio(cfg, next_state.cpu().numpy(), phase)
            else:
                audio = feature_to_audio(cfg, next_state.numpy(), phase)

            if LO:
                reward = get_score_gmm(cfg, ctime, audio, meng, feature_type, 'eval', threshold)
            else:
                score = get_score_gmm(cfg, ctime, audio, meng, feature_type)
                reward = score - base_score
                base_score = score              
            
            episode_reward += reward
            if LO:
                mask = torch.Tensor([not reward])
                print(iteration+1, 'Reward', reward)
            else:
                mask = torch.Tensor([not (score>threshold)])           
                print(iteration+1, 'Score', score, 'Reward', reward)        

            if torch.cuda.is_available():
                memory.push(state.squeeze().cpu(), action.cpu(), mask, next_state.cpu(), torch.Tensor([reward]))
            else:
                memory.push(state.squeeze(), action, mask, next_state, torch.Tensor([reward]))

            success_flag = reward if LO else score > threshold
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
                    base_audio = feature_to_audio(cfg, state.squeeze().cpu().numpy(), phase)
                else:
                    base_audio = feature_to_audio(cfg, state.squeeze().numpy(), phase)

                base_score = get_score_gmm(cfg, ctime, base_audio, meng, feature_type)
                if base_score > threshold:
                    print('False accept case. Skip this one.')
                    val_count_file += 1
                    val_feat_list.remove(val_filename+'.npy')
                    continue

                while iteration < cfg['ITER_PER_UTT']:
                    action = agent.select_action(state)
                    next_state = torch.add(state.squeeze(), action)

                    if torch.cuda.is_available():
                        audio = feature_to_audio(cfg, next_state.cpu().numpy(), phase)
                    else:
                        audio = feature_to_audio(cfg, next_state.numpy(), phase)
                    
                    score = get_score_gmm(cfg, ctime, audio, meng, feature_type)
                    reward = score - base_score
                    base_score = score
                    
                    episode_reward += reward
                    print(iteration+1, 'Score', score, 'Reward', reward)

                    if score > threshold:
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
                
            print('Average validation reward at episode {}: {}.'.format(str(episode+1), str(average_val_reward/cfg['VAL_BS'])))
            print('Validation success rate: {}/{}.'.format(str(val_success), cfg['VAL_BS']))
                    
        episode += 1
        count_file += 1
    
    meng.quit()

def evaluate(cfg, ctime, feature_type, load_actor_path=None, load_critic_path=None):
    print(torch.cuda.is_available())
    feat_dir = cfg['DATA_DIR']+'features/logmel_attack/eval/'
    phase_dir = cfg['DATA_DIR']+'features/phase_attack/eval/'
    feature_extractor_attack(cfg, 'eval')
    feat_list = os.listdir(feat_dir)
    random.shuffle(feat_list)

    if not os.path.exists(cfg['ROOT_DIR']+'evading_audio/{}/'.format(ctime)):
        os.system('mkdir -p '+cfg['ROOT_DIR']+'evading_audio/{}/'.format(ctime))
    if not os.path.exists(cfg['ROOT_DIR']+'temp_audio/'):
        os.system('mkdir -p '+cfg['ROOT_DIR']+'temp_audio/')

    agent = DDPG(cfg, feature_type)
    agent.load_model(load_actor_path, load_critic_path)
    agent.save_model(ctime, 0)
    rewards = []
    count = 0
    count_file = 0
    success_queries = 0
    
    meng = matlab.engine.start_matlab()
    if feature_type == 'LFCC':
        threshold = 0.748917
    elif feature_type == 'CQCC':
        threshold = 1.252953
    else:
        print('Feature type not available.') 

    while count < cfg['EVAL_NUM']:
        filename = feat_list[count_file][:-4]
        state = np.expand_dims(np.load(feat_dir+feat_list[count_file]), axis=0)
        state = torch.from_numpy(state).to(device)
        phase = np.load(phase_dir+feat_list[count_file])
        episode_reward = 0
        iteration = 0

        print('Evaluation', count+1, filename)
        if torch.cuda.is_available():    
            base_audio = feature_to_audio(cfg, state.squeeze().cpu().numpy(), phase)
        else:
            base_audio = feature_to_audio(cfg, state.squeeze().numpy(), phase)

        base_reward = get_score_gmm(cfg, ctime, base_audio, meng, feature_type, 'eval', threshold)
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
                audio = feature_to_audio(cfg, next_state.cpu().numpy(), phase)
            else:
                audio = feature_to_audio(cfg, next_state.numpy(), phase)

            reward = get_score_gmm(cfg, ctime, audio, meng, feature_type, 'eval', threshold)
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
        if episode_reward:
            success_queries += (iteration+1)
            print('Current average queries:', float(success_queries)/float(sum(rewards)))
    
    meng.quit()

    print('Successful rate: {}/{}.'.format(sum(rewards), cfg['EVAL_NUM']))