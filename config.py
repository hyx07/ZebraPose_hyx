import argparse

class TrainArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(TrainArgumentParser, self).__init__(*args, **kwargs)
        #dataset
        self.add_argument('--bop_path',type=str,help='bop top path')
        self.add_argument('--dataset_name',type=str,help='dataset name, e.g. lm')
        self.add_argument('--training_data_folder',type=str,help='folder containing train data')
        self.add_argument('--training_data_folder2',type=str,help='second folder containing train data')
        self.add_argument('--second_dataset_ratio', type=float, default=0.75)
        self.add_argument('--val_folder',type=str,help='validation data folder')
        self.add_argument('--train_obj_visible_threshold',type=float,default=0.2,help='threshold for selecting training data')
        
        self.add_argument('--obj_name',type=str,help='object name for training')
        
        self.add_argument('--bbox_cropsize_img',type=int,default=256,help='crop size from image')
        self.add_argument('--bbox_cropsize_gt',type=int,default=128,help='network output size')
        
        #net
        self.add_argument('--resnet_layer',type=int,default=34,help='resnet layer num')
        self.add_argument('--concat_encoder_decoder',type=bool,default=True,help='if use concat or not')
        self.add_argument('--binary_code_length',type=int,default=16)
        self.add_argument('--divided_number_each_iteration',type=int,default=2)
        self.add_argument('--output_kernel_size',type=int,default=1)
        self.add_argument('--efficientnet_key',type=str,default=None)

        #loss
        self.add_argument('--BinaryCode_Loss_Type',type=str,default='BCE')
        self.add_argument('--mask_binary_code_loss',type=bool,default=True)
        self.add_argument('--use_histgramm_weighted_binary_loss',type=bool,default=True)
        self.add_argument('--binary_loss_weight',type=float,default=3.0)

        #train_setup
        self.add_argument('--is_train',type=bool,default=True)
        self.add_argument('--gpu_ids',type=str,default='0')
        self.add_argument('--num_workers',type=int,default=1)
        self.add_argument('--continue_train_epoch',type=str,default=None)
        
        self.add_argument('--optimizer_type',type=str,default='Adam')
        self.add_argument('--learning_rate',type=float,default=0.0002)
        self.add_argument('--batch_size',type=int,default=2)
        
        self.add_argument('--epoch_count',type=int,default=1)
        self.add_argument('--n_epochs',type=int,default=1)
        self.add_argument('--n_epochs_decay',type=int,default=1)
        
        
        #augmentations
        self.add_argument('--padding_ratio',type=float,default=1.5)
        self.add_argument('--resize_method',type=str,default='crop_square_resize')
        self.add_argument('--use_peper_salt',type=bool,default=True)
        self.add_argument('--use_motion_blur',type=bool,default=True)
        self.add_argument('--sym_aware_training',type=bool,default=False)

        #save
        self.add_argument('--save_dir',type=str,default='/opao')
        self.add_argument('--save_epoch_interval',type=int,default=100)
        self.add_argument('--test_ADD_epoch_interval',type=int,default=25)
        self.add_argument('--train_ADD_iter_interval',type=int,default=1000)
        self.add_argument('--test_loss_epoch_interval',type=int,default=1)
        self.add_argument('--print_loss_interval',type=int,default=10)


class TestArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(TestArgumentParser, self).__init__(*args, **kwargs)
        #dataset
        self.add_argument('--bop_path',type=str,help='bop top path')
        self.add_argument('--dataset_name',type=str,help='dataset name, e.g. lm')
        self.add_argument('--training_data_folder',type=str,help='folder containing train data')
        self.add_argument('--training_data_folder2',type=str,help='second folder containing train data')
        self.add_argument('--second_dataset_ratio', type=float, default=0.75)
        self.add_argument('--val_folder',type=str,help='validation data folder')
        self.add_argument('--train_obj_visible_threshold',type=float,default=0.2,help='threshold for selecting training data')
        
        self.add_argument('--obj_name',type=str,help='object name for training')
        
        self.add_argument('--bbox_cropsize_img',type=int,default=256,help='crop size from image')
        self.add_argument('--bbox_cropsize_gt',type=int,default=128,help='network output size')
        
        #net
        self.add_argument('--resnet_layer',type=int,default=34,help='resnet layer num')
        self.add_argument('--concat_encoder_decoder',type=bool,default=True,help='if use concat or not')
        self.add_argument('--binary_code_length',type=int,default=16)
        self.add_argument('--divided_number_each_iteration',type=int,default=2)
        self.add_argument('--output_kernel_size',type=int,default=1)
        self.add_argument('--efficientnet_key',type=str,default=None)

        #test_setup
        self.add_argument('--is_train',type=bool,default=False)
        self.add_argument('--gpu_ids',type=str,default='0')
        self.add_argument('--num_workers',type=int,default=8)
        self.add_argument('--batch_size',type=int,default=2)

        self.add_argument('--BinaryCode_Loss_Type',type=str,default='BCE')

        self.add_argument('--eval_with_ignore_bits',type=bool,default=False)
        #data
        self.add_argument('--padding_ratio',type=float,default=1.5)
        self.add_argument('--resize_method',type=str,default='crop_resize')
        self.add_argument('--use_peper_salt',type=bool,default=True)
        self.add_argument('--use_motion_blur',type=bool,default=True)
        self.add_argument('--sym_aware_training',type=bool,default=False)

        #save
        self.add_argument('--load_dir',type=str,default='/zebra')
        self.add_argument('--load_epoch',type=str,default='best')
        
