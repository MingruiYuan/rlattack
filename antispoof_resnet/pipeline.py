import os, time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np 
import matplotlib.pyplot as plt
from data.asvset import AntispoofingSet
from antispoof_resnet.models import MFCCModel

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def collate_dict(batch):
    feats = torch.stack([batch[i]['FEAT'] for i in range(len(batch))], dim=0)
    labels = torch.stack([batch[i]['LABEL'] for i in range(len(batch))], dim=0)
    return {'FEATS': feats, 'LABELS': labels}

def init_weights(layer):
    if hasattr(layer, 'weight'):
        if len(layer.weight.shape) > 1:
            torch.nn.init.kaiming_normal_(layer.weight)

def evaluate_accuracy(data_loader, model, criterion):
    print('Validation...')
    running_loss = 0
    # n_correct = 0.0
    n_total = len(data_loader)
    scores_bonafide = []
    scores_spoof = []
    thres = np.linspace(-10,10,101) 

    model.eval()
    for i, dt in enumerate(data_loader):
        # if i % 10 == 0:
        #     print('Iteration of validation {}/{}'.format(str(i+1), str(len(data_loader))))
        feats = dt['FEATS'].to(device)
        labels = dt['LABELS'].view(-1).long().to(device)
        outputs = model(feats)
        loss = criterion(outputs, labels)
        if i % 1000 == 0:
            print('label {}, loss {}, lpd {}'.format(labels.item(), loss.item(), outputs[0][1].item()-outputs[0][0].item()))
        if labels.item():
            scores_bonafide.append(outputs[0][1].item()-outputs[0][0].item())
        else:
            scores_spoof.append(outputs[0][1].item()-outputs[0][0].item())
        # _, pred_labels = outputs.max(dim=1)
        # n_correct += (pred_labels == labels).sum(dim=0).item()
        running_loss += loss.item()
        # if i % 10 == 0:
            # print('Accumulated accuracy: ', n_correct/n_total)

    n_bonafide = len(scores_bonafide)
    n_spoof = len(scores_spoof)
    flag1 = 1
    flag2 = 1
    EER = 0
    threshold = 0
    scores_bonafide = torch.Tensor(scores_bonafide)
    scores_spoof = torch.Tensor(scores_spoof)
    for i in range(len(thres)):
        FRR = (scores_bonafide<thres[i]).sum().float().item()/n_bonafide
        FAR = (scores_spoof>thres[i]).sum().float().item()/n_spoof
        flag2 = flag1
        flag1 = (FAR - FRR) > 0
        if flag1 != flag2:
            EER = (FAR+FRR)/2
            threshold = thres[i]
            break
        print('Threshold {}: FAR {}, FRR {}.'.format(str(thres[i]), str(FAR), str(FRR)))

    return running_loss/n_total, threshold, EER

def plot_curves(losses, epoch, fig_dir):
    if not os.path.exists(fig_dir):
        os.system('mkdir -p '+fig_dir)

    fig1, ax1 = plt.subplots(2,1)
    lenval = len(losses['vllog'])
    ax1[0].set_title('Validation Loss')
    ax1[1].set_title('Validation Accuracy')
    ax1[0].plot(np.linspace(1,lenval,lenval), losses['vllog'], 'r--')
    ax1[1].plot(np.linspace(1,lenval,lenval), losses['valog'], 'g')
    plt.savefig(fig_dir+'ValCurve_iteration_{}.png'.format(str(epoch)), format='png')

    fig2, ax2 = plt.subplots(2,1)
    lentrn= len(losses['tllog'])
    ax2[0].set_title('Training Loss')
    ax2[1].set_title('Training Accuracy')
    ax2[0].plot(np.linspace(1,lentrn,lentrn), losses['tllog'], 'r--')
    ax2[1].plot(np.linspace(1,lentrn,lentrn), losses['talog'], 'g')
    plt.savefig(fig_dir+'TrnCurve_iteration_{}.png'.format(str(epoch)), format='png')

def train(cfg, ctime, load_path=None):
    trset = AntispoofingSet(cfg, 'train')
    trloader = DataLoader(trset, batch_size=cfg['ASV_BS'], shuffle=True, num_workers=2, collate_fn=collate_dict)
    devset = AntispoofingSet(cfg, 'dev')
    devloader = DataLoader(devset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_dict)

    weight = torch.FloatTensor([1.0, 9.0]).to(device)
    criterion = nn.NLLLoss(weight=weight)
    
    model = MFCCModel()
    if load_path is not None:
        ckpt = torch.load(load_path)
        model.load_state_dict(ckpt['model_state_dict'])
        epoch = ckpt['epoch']
        model.to(device)
        optimizer = optim.Adam(model.parameters(), cfg['ASV_LR'])
        optimizer.load_state_dict(ckpt['opt_state_dict'])
        # trn_loss_log = ckpt['tllog']
        # val_loss_log = ckpt['vllog']
        # trn_acc_log = ckpt['talog']
        # val_acc_log = ckpt['valog']
    else:
        model.apply(init_weights)
        epoch = 0
        model.to(device)
        optimizer = optim.Adam(model.parameters(), cfg['ASV_LR'])
        # trn_loss_log = []
        # val_loss_log = []
        # trn_acc_log = []
        # val_acc_log = []

    while epoch < cfg['ASV_MAXEPOCH']:
        running_loss = 0
        # n_correct = 0.0
        n_total = 0.0
        model.train()
        for i, dt in enumerate(trloader):
            if i % 100 == 0:
                print('Epoch {}, Iteration {}/{}'.format(str(epoch+1), str(i+1), str(len(trloader))))
            optimizer.zero_grad()
            feats = dt['FEATS'].to(device)
            labels = dt['LABELS'].view(-1).long().to(device)
            n_total += feats.shape[0]

            outputs = model(feats)
            loss = criterion(outputs, labels)
            # _, pred_labels = outputs.max(dim=1)
            # n_correct += (pred_labels == labels).sum(dim=0).item()
            running_loss += (loss.item()*feats.shape[0])
            # if i % 10 == 0:
            #     print('Accumulated accuracy: ', n_correct/n_total)
            loss.backward()
            optimizer.step()
            # trn_loss_log.append(loss.item())
            # trn_acc_log.append(n_correct/n_total)

        running_loss /= n_total
        # training_acc = n_correct/n_total # Should we compute EER ??
        print('Epoch {}, training loss:{}, '.format(str(epoch+1), str(running_loss)))
        val_loss, thres, EER = evaluate_accuracy(devloader, model, criterion)
        # val_loss_log.append(val_loss)
        # val_acc_log.append(val_acc)
        print('Epoch {}, validation loss:{}, threshold:{}, EER:{}'.format(str(epoch+1), str(val_loss), str(thres), str(EER)))

        save_path = cfg['ROOT_DIR'] +'saved_models/antispoof_resnet/{}/'.format(ctime)
        if not os.path.exists(save_path):
            os.system('mkdir -p '+save_path)

        # fig_path = save_path + 'fig/'
        # curves = {'tllog':trn_loss_log, 'talog':trn_acc_log, 'vllog':val_loss_log, 'valog':val_acc_log}
        # plot_curves(curves, epoch+1, fig_path)

        # Also need to plot curves.
        torch.save({'epoch': epoch+1,
            'model_state_dict': model.state_dict(),
            'opt_state_dict': optimizer.state_dict()
            # 'tllog': trn_loss_log,
            # 'vllog': val_loss_log,
            # 'talog': trn_acc_log,
            # 'valog': val_acc_log
            }, save_path+'resnet_mfcc_epoch_{}.pth'.format(str(epoch+1)))

        epoch += 1

def evaluate(cfg):
    evalset = AntispoofingSet(cfg, 'eval')
    evalloader = DataLoader(evalset, batch_size=1, shuffle=False, num_workers=2, collate_fn=collate_dict)
    model = MFCCModel()
    load_path = cfg['ROOT_DIR'] + 'saved_models/antispoof_resnet/20-01-16_08-03-57/resnet_mfcc_epoch_50.pth'
    print('Model: ', load_path)
    ckpt = torch.load(load_path)
    model.load_state_dict(ckpt['model_state_dict'])

    scores_bonafide = []
    scores_spoof = []
    model.eval()

    weight = torch.FloatTensor([1.0, 9.0]).to(device)
    criterion = nn.NLLLoss(weight=weight)

    with torch.no_grad():
        for i, dt in enumerate(evalloader):
            print('{}/{} evaluation.'.format(str(i+1), str(len(evalloader))))
            feats = dt['FEATS'].to(device)
            labels = dt['LABELS'].view(-1).long().to(device)
            outputs = model(feats)
            loss = criterion(outputs, labels)
            print('label {}, loss {}, lpd {}'.format(labels.item(), loss.item(), outputs[0][1].item()-outputs[0][0].item()))
            if labels.item():
                scores_bonafide.append(outputs[0][1].item()-outputs[0][0].item())
            else:
                scores_spoof.append(outputs[0][1].item()-outputs[0][0].item())

    n_bonafide = len(scores_bonafide)
    n_spoof = len(scores_spoof)
    print('{} bonafide samples, {} spoof samples.'.format(n_bonifide, n_spoof))
    scores_bonafide = torch.Tensor(scores_bonafide)
    scores_spoof = torch.Tensor(scores_spoof)
    thres = np.linspace(-10,10,101)
    for i in range(len(thres)):
        FRR = (scores_bonafide<thres[i]).sum().float().item()/n_bonafide
        FAR = (scores_spoof>thres[i]).sum().float().item()/n_spoof
        print('Threshold {}: FAR {}, FRR {}.'.format(str(thres[i]), str(FAR), str(FRR)))