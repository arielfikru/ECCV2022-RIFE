import os
import cv2
import math
import time
import torch
import numpy as np
import random
import argparse

from model.RIFE import Model
from dataset import *
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log_path = 'train_log'

def get_learning_rate(step, total_steps):
    if step < 2000:
        mul = step / 2000.
        return 3e-4 * mul
    else:
        mul = np.cos((step - 2000) / (total_steps - 2000.) * math.pi) * 0.5 + 0.5
        return (3e-4 - 3e-6) * mul + 3e-6

def flow2rgb(flow_map_np):
    h, w, _ = flow_map_np.shape
    rgb_map = np.ones((h, w, 3)).astype(np.float32)
    normalized_flow_map = flow_map_np / (np.abs(flow_map_np).max())
    
    rgb_map[:, :, 0] += normalized_flow_map[:, :, 0]
    rgb_map[:, :, 1] -= 0.5 * (normalized_flow_map[:, :, 0] + normalized_flow_map[:, :, 1])
    rgb_map[:, :, 2] += normalized_flow_map[:, :, 1]
    return rgb_map.clip(0, 1)

def train(model):
    writer = SummaryWriter('train')
    writer_val = SummaryWriter('validate')
    step = 0
    nr_eval = 0
    
    # Initialize datasets without distributed sampling
    dataset = VimeoDataset('train')
    train_data = DataLoader(dataset, batch_size=args.batch_size, num_workers=2, 
                          pin_memory=True, drop_last=True, shuffle=True)
    
    dataset_val = VimeoDataset('validation')
    val_data = DataLoader(dataset_val, batch_size=batch_size, pin_memory=True, num_workers=2)
    
    args.step_per_epoch = len(train_data)
    total_steps = args.epoch * args.step_per_epoch
    
    print('Training started...')
    time_stamp = time.time()
    
    for epoch in range(args.epoch):
        for i, data in enumerate(train_data):
            data_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            
            data_gpu, timestep = data
            data_gpu = data_gpu.to(device, non_blocking=True) / 255.
            timestep = timestep.to(device, non_blocking=True)
            
            imgs = data_gpu[:, :6]
            gt = data_gpu[:, 6:9]
            
            learning_rate = get_learning_rate(step, total_steps)
            pred, info = model.update(imgs, gt, learning_rate, training=True)
            
            train_time_interval = time.time() - time_stamp
            time_stamp = time.time()
            
            if step % 200 == 1:
                writer.add_scalar('learning_rate', learning_rate, step)
                writer.add_scalar('loss/l1', info['loss_l1'], step)
                writer.add_scalar('loss/tea', info['loss_tea'], step)
                writer.add_scalar('loss/distill', info['loss_distill'], step)
                
            if step % 1000 == 1:
                gt = (gt.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                mask = (torch.cat((info['mask'], info['mask_tea']), 3).permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                pred = (pred.permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                merged_img = (info['merged_tea'].permute(0, 2, 3, 1).detach().cpu().numpy() * 255).astype('uint8')
                flow0 = info['flow'].permute(0, 2, 3, 1).detach().cpu().numpy()
                flow1 = info['flow_tea'].permute(0, 2, 3, 1).detach().cpu().numpy()
                
                for i in range(min(5, len(pred))):
                    imgs = np.concatenate((merged_img[i], pred[i], gt[i]), 1)[:, :, ::-1]
                    writer.add_image(str(i) + '/img', imgs, step, dataformats='HWC')
                    writer.add_image(str(i) + '/flow', np.concatenate((flow2rgb(flow0[i]), flow2rgb(flow1[i])), 1), step, dataformats='HWC')
                    writer.add_image(str(i) + '/mask', mask[i], step, dataformats='HWC')
                writer.flush()
                
            print('epoch:{} {}/{} time:{:.2f}+{:.2f} loss_l1:{:.4e}'.format(
                epoch, i, args.step_per_epoch, data_time_interval, train_time_interval, info['loss_l1']))
            step += 1
            
        nr_eval += 1
        if nr_eval % 5 == 0:
            evaluate(model, val_data, step, writer_val)
            
        model.save_model(log_path)
        
def evaluate(model, val_data, nr_eval, writer_val):
    loss_l1_list = []
    loss_distill_list = []
    loss_tea_list = []
    psnr_list = []
    psnr_list_teacher = []
    time_stamp = time.time()
    
    for i, data in enumerate(val_data):
        data_gpu, timestep = data
        data_gpu = data_gpu.to(device, non_blocking=True) / 255.
        imgs = data_gpu[:, :6]
        gt = data_gpu[:, 6:9]
        
        with torch.no_grad():
            pred, info = model.update(imgs, gt, training=False)
            merged_img = info['merged_tea']
            
        loss_l1_list.append(info['loss_l1'].cpu().numpy())
        loss_tea_list.append(info['loss_tea'].cpu().numpy())
        loss_distill_list.append(info['loss_distill'].cpu().numpy())
        
        for j in range(gt.shape[0]):
            psnr = -10 * math.log10(torch.mean((gt[j] - pred[j]) * (gt[j] - pred[j])).cpu().data)
            psnr_list.append(psnr)
            psnr = -10 * math.log10(torch.mean((merged_img[j] - gt[j]) * (merged_img[j] - gt[j])).cpu().data)
            psnr_list_teacher.append(psnr)
            
        if i == 0:
            gt = (gt.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
            pred = (pred.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
            merged_img = (merged_img.permute(0, 2, 3, 1).cpu().numpy() * 255).astype('uint8')
            flow0 = info['flow'].permute(0, 2, 3, 1).cpu().numpy()
            
            for j in range(min(10, len(pred))):
                imgs = np.concatenate((merged_img[j], pred[j], gt[j]), 1)[:, :, ::-1]
                writer_val.add_image(str(j) + '/img', imgs.copy(), nr_eval, dataformats='HWC')
                writer_val.add_image(str(j) + '/flow', flow2rgb(flow0[j][:, :, ::-1]), nr_eval, dataformats='HWC')
    
    writer_val.add_scalar('psnr', np.array(psnr_list).mean(), nr_eval)
    writer_val.add_scalar('psnr_teacher', np.array(psnr_list_teacher).mean(), nr_eval)

if __name__ == "__main__":    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument('--batch_size', default=4, type=int, help='minibatch size')
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    seed = 1234
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True
    
    # Initialize model
    model = Model()
    train(model)
