import argparse

class inferenceArgumentParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super(inferenceArgumentParser, self).__init__(*args, **kwargs)
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
        self.add_argument('--gpu_ids',type=str,default='0')
        
        #pre_process
        self.add_argument('--padding_ratio',type=float,default=1.5)
        self.add_argument('--resize_method',type=str,default='crop_square_resize')

        #self.add_argument('--cam_K',type=list,default=[616.58,0,323.103, 0,616.778,238.464, 0.0, 0.0, 1.0])
        
##        #paths
##        self.add_argument('--model_path',type=str,default='/zebrapose/assets/model.pth')
##        self.add_argument('--model_dict_path',type=str,default='/zebrapose/assets/Class_CorresPoint000001.txt')
