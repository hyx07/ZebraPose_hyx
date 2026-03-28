import torch
import torch.nn as nn
import os
import numpy as np

from networks.BinaryCodeNet import BinaryCodeLoss, MaskLoss
from utils import from_output_to_class_mask, from_output_to_class_binary_code

class Zebra():
    def __init__(self, opt, optimizer):
        self.opt = opt

        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.n_epochs) / float(opt.n_epochs_decay + 1)
            return lr_l

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)

        self.binarycode_loss = BinaryCodeLoss(
            opt.BinaryCode_Loss_Type,
            opt.mask_binary_code_loss,
            opt.divided_number_each_iteration,
            use_histgramm_weighted_binary_loss=opt.use_histgramm_weighted_binary_loss
            )

        self.mask_loss = MaskLoss()
        self.binary_loss_weight = opt.binary_loss_weight
        
    def set_input_data(self, inputs, masks, binary_target, device):
        self.inputs = inputs.to(device)
        self.mask_target = masks.to(device)
        self.binary_target = binary_target.to(device)

    def optimize_parameters(self, zebra_net, optimizer, device):
        zebra_net.train()
        optimizer.zero_grad()
        pred_mask_prob, pred_code_prob = zebra_net(self.inputs)

        pred_mask_for_loss = from_output_to_class_mask(pred_mask_prob)
        pred_mask_for_loss = torch.tensor(pred_mask_for_loss).to(device)
        self.loss_b = self.binarycode_loss(pred_code_prob, pred_mask_for_loss, self.binary_target)

        self.loss_m = self.mask_loss(pred_mask_prob, self.mask_target)
        
        self.loss = self.binary_loss_weight*self.loss_b + self.loss_m
        self.loss.backward()
        optimizer.step()

    def evaluation(self, test_dataloader, zebra_net, device):
        test_loss = [[],[],[]]
        zebra_net.eval()
        for batch_idx, (data, entire_masks, obj_masks, Rs, ts, Bboxes, class_code_images, cam_Ks) in enumerate(test_dataloader):
            inputs, masks,  binary_codes = data, obj_masks, class_code_images
            with torch.no_grad():
                pred_mask_prob, pred_code_prob = zebra_net(inputs.to(device))

            pred_mask_for_loss = from_output_to_class_mask(pred_mask_prob)
            pred_mask_for_loss = torch.tensor(pred_mask_for_loss).to(device)
            loss_b = self.binarycode_loss(pred_code_prob, pred_mask_for_loss, binary_codes.to(device), False)
            loss_m = self.mask_loss(pred_mask_prob, masks.to(device))
            loss_total = self.binary_loss_weight*loss_b + loss_m
            test_loss[0].append(loss_total.item())
            test_loss[1].append(loss_b.item())
            test_loss[2].append(loss_m.item())
        return test_loss
    
    def return_loss(self):
        return self.loss.item(), self.loss_b.item(), self.loss_m.item()
    
    def update_learning_rate(self, optimizer):
        self.scheduler.step()
        lr = optimizer.param_groups[0]['lr']
        print('current lr = %.7f'%lr, flush=True)

    def save_networks(self, epoch, zebra_net, optimizer):
        if epoch == 'latest':
            save_path = './checkpoints'+self.opt.save_dir+'/latest.pth'
        elif epoch == 'best':
            save_path = './checkpoints'+self.opt.save_dir+'/best.pth'
        else:
            save_path = './checkpoints'+self.opt.save_dir+'/%s.pth'%epoch
            
        torch.save(
            {'net_state_dict':zebra_net.module.state_dict(),
             'optimizer_state_dict':optimizer.state_dict(),
             'epoch':epoch},
            save_path)

    def save_snapshot(self, epoch, zebra_net, optimizer, save_path):   
        torch.save(
            {'net_state_dict':zebra_net.module.state_dict(),
             'optimizer_state_dict':optimizer.state_dict(),
             'epoch':epoch},
            save_path)
        
    def train_continue_load(self, zebra_net, optimizer):
        load_path = './checkpoints'+self.opt.save_dir+'/%s.pth'%self.opt.continue_train_epoch
        loaded_model = torch.load(load_path)
        zebra_net.module.load_state_dict(loaded_model['net_state_dict'])
        optimizer.load_state_dict(loaded_model['optimizer_state_dict'])
        print('model loaded:', load_path, flush=True)
