import config
import model
import time
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import os

from networks.BinaryCodeNet import BinaryCodeNet_Deeplab
from utils import evaluate_model, evaluate_train_model, get_train_dataloader, get_test_dataloader, save_options

class Trainer():
    def __init__(self, zebra_net, optimizer, model_wrapper, train_dataloader, obj_info, test_dataloader, writer, opt):
        self.gpu_id = int(os.environ["LOCAL_RANK"])
        self.zebra_net = zebra_net.to(self.gpu_id)
        self.zebra_net = DDP(self.zebra_net, device_ids=[self.gpu_id])
        self.optimizer = optimizer
        self.model_wrapper = model_wrapper
        self.train_dataloader = train_dataloader
        self.obj_info = obj_info
        self.test_dataloader = test_dataloader
        self.writer = writer
        self.opt = opt
        self.epoch_count = self.opt.epoch_count

        self.snapshot_path = './checkpoints'+self.opt.save_dir+'/snapshot.pth'
        if os.path.exists(self.snapshot_path):
            self._load_snapshot()

        if self.opt.continue_train_epoch is not None:
            self.model_wrapper.train_continue_load(self.zebra_net, self.optimizer)

    def train(self):
        if self.gpu_id == 0:
            self.best_ADD = -1.0
            save_options(self.opt, self.best_ADD)
        print('training start!', flush=True)
        for epoch in range(self.epoch_count, self.opt.n_epochs + self.opt.n_epochs_decay + 1):
            self.run_epoch(epoch)

        print('training done!', flush=True)
        print('*'*30, flush=True)
        
        if self.gpu_id == 0:
            print('best score:', self.best_ADD, flush=True)
            save_options(self.opt, self.best_ADD)
            
    def run_epoch(self, epoch):
        self.train_dataloader.sampler.set_epoch(epoch)
        epoch_start = time.time()
        epoch_loss = [[],[],[]]
        for batch_idx, (data, entire_masks, obj_masks, Rs, ts, Bboxes, class_code_images, cam_Ks) in enumerate(self.train_dataloader):
            loss = self.run_batch(data, obj_masks, class_code_images)
            epoch_loss[0].append(loss[0])
            epoch_loss[1].append(loss[1])
            epoch_loss[2].append(loss[2])

            if batch_idx % self.opt.print_loss_interval == 0:
                print('[GPU%d]epoch:%02d/%02d | iter:%04d/%04d | loss_total:%.4f | loss_binary:%.4f | loss_mask:%.4f'%
                      (self.gpu_id, epoch, (self.opt.n_epochs + self.opt.n_epochs_decay), batch_idx, len(self.train_dataloader), loss[0], loss[1], loss[2]), flush=True)

            iter_num = batch_idx + (epoch-1)*len(self.train_dataloader) + 1
            if iter_num%self.opt.train_ADD_iter_interval == 0:
                train_ADD_passed, train_ADD_error, train_AUC_ADD_error = evaluate_train_model(self.model_wrapper, self.zebra_net, self.obj_info, self.opt, data, obj_masks, Rs, ts, Bboxes, cam_Ks, 0, self.gpu_id)
                print('[GPU%d]train data ADD:'%self.gpu_id, train_ADD_passed, flush=True)
                self.writer.add_scalar('TRAINDATA_ADD/ADD_test GPU%d'%self.gpu_id, train_ADD_passed, epoch)
                self.writer.add_scalar('TRAINDATA_ADD/ADD_Error_test GPU%d'%self.gpu_id, train_ADD_error, epoch)
                self.writer.add_scalar('TRAINDATA_AUC_ADD/AUC_ADD_Error_test GPU%d'%self.gpu_id, train_AUC_ADD_error, epoch)

        epoch_duration = time.time() - epoch_start
        print('[GPU%d]epoch training time:%.0f'%(self.gpu_id, epoch_duration), flush=True)
        print('[GPU%d]epoch loss_total:%.4f'%(self.gpu_id, np.mean(epoch_loss[0])), flush=True)

        self.writer.add_scalar('Loss/train loss total GPU%d'%self.gpu_id, np.mean(epoch_loss[0]), epoch)
        self.writer.add_scalar('Loss/train loss binary GPU%d'%self.gpu_id, np.mean(epoch_loss[1]), epoch)
        self.writer.add_scalar('Loss/train loss mask GPU%d'%self.gpu_id, np.mean(epoch_loss[2]), epoch)

        if self.gpu_id == 0 and epoch % self.opt.save_epoch_interval == 0:
            self.model_wrapper.save_networks(epoch, self.zebra_net, self.optimizer)

        if self.gpu_id == 0 and epoch%self.opt.test_loss_epoch_interval == 0:
            print('evaluate model:', flush=True)
            test_loss = self.model_wrapper.evaluation(self.test_dataloader, self.zebra_net, self.gpu_id)
            print('evaluation total loss:', np.mean(test_loss[0]), np.mean(test_loss[1]), np.mean(test_loss[2]), flush=True)

            self.writer.add_scalar('Loss/test loss total', np.mean(test_loss[0]), epoch)
            self.writer.add_scalar('Loss/test loss binary', np.mean(test_loss[1]), epoch)
            self.writer.add_scalar('Loss/test loss mask', np.mean(test_loss[2]), epoch)        

        if self.gpu_id == 0 and epoch%self.opt.test_ADD_epoch_interval == 0:
            print('*'*20, flush=True)
            print('evaluating the model...', flush=True)
            ADD_passed, ADD_error, AUC_ADD_error = evaluate_model(self.model_wrapper, self.zebra_net, self.obj_info, self.test_dataloader, self.writer, self.opt, epoch, self.gpu_id, 0, calc_add_and_adi=False)
            print('evaluation ADD:', ADD_passed, flush=True)
            print('evaluation ADD error:', ADD_error, flush=True)
            print('evaluation AUC ADD error:', AUC_ADD_error, flush=True)

            if ADD_passed > self.best_ADD:
                self.best_ADD = ADD_passed
                self.model_wrapper.save_networks('best', self.zebra_net, self.optimizer)
        
        if self.gpu_id == 0:
            self.model_wrapper.save_snapshot(epoch, self.zebra_net, self.optimizer, self.snapshot_path)
            print('*'*30, flush=True)
        self.model_wrapper.update_learning_rate(self.optimizer)
        
    def run_batch(self, inputs, obj_masks, class_code_images):
        self.model_wrapper.set_input_data(inputs, obj_masks, class_code_images, self.gpu_id)
        self.model_wrapper.optimize_parameters(self.zebra_net, self.optimizer, self.gpu_id)
        loss = self.model_wrapper.return_loss()
        return loss

    def _load_snapshot(self):
        snapshot = torch.load(self.snapshot_path)
        self.zebra_net.module.load_state_dict(snapshot['net_state_dict'])
        self.optimizer.load_state_dict(snapshot['optimizer_state_dict'])
        self.epoch_count = int(snapshot['epoch']) + 1
        print('Resuming training from epoch:%d'%(self.epoch_count-1), flush=True)
    
def load_train_obj(opt):
    #get model
    zebra_net = BinaryCodeNet_Deeplab(
        num_resnet_layers=opt.resnet_layer,
        concat = opt.concat_encoder_decoder,
        binary_code_length=opt.binary_code_length,
        divided_number_each_iteration=opt.divided_number_each_iteration,
        output_kernel_size=opt.output_kernel_size,
        efficientnet_key=opt.efficientnet_key)
    print('model created!', flush=True)

    if opt.optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(zebra_net.parameters(), lr=opt.learning_rate, betas=(0.9, 0.999))
    elif opt.optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(zebra_net.parameters(), lr=opt.learning_rate, momentum=0.9)

    model_wrapper = model.Zebra(opt, optimizer)
    
    train_dataloader, obj_info = get_train_dataloader(opt, opt.training_data_folder, opt.batch_size)
    print('dataloader created!', flush=True)

    test_dataloader = get_test_dataloader(opt, obj_info[0])

    #create logger
    writer = SummaryWriter('./checkpoints'+opt.save_dir+'/tensorboard/')
    return zebra_net, optimizer, model_wrapper, train_dataloader, obj_info, test_dataloader, writer

def ddp_setup():
    init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    
def main(opt):
    ddp_setup()
    zebra_net, optimizer, model_wrapper, train_dataloader, obj_info, test_dataloader, writer = load_train_obj(opt)
    trainer = Trainer(zebra_net, optimizer, model_wrapper, train_dataloader, obj_info, test_dataloader, writer, opt)
    trainer.train()
    destroy_process_group()


if __name__ == '__main__':
    parser = config.TrainArgumentParser()
    opt = parser.parse_args()

    opt.bop_path = '/path to dataset here'
    opt.dataset_name = 'dataset name here'
    opt.training_data_folder = 'data folder name in dataset folder'
    opt.val_folder = 'valuation data folder name in dataset folder'
    opt.obj_name = 'target object name'

    main(opt)


        
