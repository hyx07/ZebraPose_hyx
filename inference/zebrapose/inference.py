import numpy as np
import torch
import cv2
import os
import time
from .config import inferenceArgumentParser

from .networks.BinaryCodeNet import BinaryCodeNet_Deeplab
from .utils import from_output_to_class_mask, from_output_to_class_binary_code, padding_Bbox, get_roi, get_final_Bbox, transform_pre
from .binary_code_helper.CNN_output_to_pose import CNN_outputs_to_object_pose, load_dict_class_id_3D_points

class zebra():
    def __init__(self, obj_name, model_id, cam_K):
        parser = inferenceArgumentParser()
        self.opt = parser.parse_args()
        self.device = torch.device('cuda:{}'.format(self.opt.gpu_ids[0])) if self.opt.gpu_ids else torch.device('cpu')
    
        self.zebra_net = BinaryCodeNet_Deeplab(
            num_resnet_layers = self.opt.resnet_layer,
            concat = self.opt.concat_encoder_decoder,
            binary_code_length = self.opt.binary_code_length,
            divided_number_each_iteration = self.opt.divided_number_each_iteration,
            output_kernel_size = self.opt.output_kernel_size,
            efficientnet_key = self.opt.efficientnet_key)

        current_dir = os.getcwd()
        #begin = time.time()
        self.zebra_net.load_state_dict(torch.load(current_dir+'/assets/%s.pth'%obj_name)['net_state_dict'])
        self.zebra_net.to(self.device)
        self.zebra_net.eval()
        #load_time = time.time() - begin
        #print('load time:', load_time)

        self.cam_K = np.array(cam_K).reshape(3,3)

        _,_,_,self.model_dict = load_dict_class_id_3D_points(current_dir+'/assets/Class_CorresPoint%06d.txt'%model_id)
    
    def predict(self, img, detected_box, debug=False):
        if debug:
            now = time.time()

        center_x, center_y, w, h = detected_box
        converted_box = np.array([int(center_x-w/2), int(center_y-h/2), int(w), int(h)])
        roi, Bbox = self.preprocess(img, converted_box)
        with torch.no_grad():
            pred_mask_prob, pred_code_prob = self.zebra_net(roi.to(self.device))

 
        pred_mask = from_output_to_class_mask(pred_mask_prob[0,0])
        pred_codes = from_output_to_class_binary_code(pred_code_prob[0], self.opt.BinaryCode_Loss_Type,
                                                      divided_num_each_iteration=self.opt.divided_number_each_iteration,
                                                      binary_code_length=self.opt.binary_code_length)

        pred_codes = pred_codes.transpose(1,2,0)
        r_predict, t_predict, success = CNN_outputs_to_object_pose(pred_mask,
                                                                  pred_codes,
                                                                  Bbox,
                                                                  self.opt.bbox_cropsize_gt,
                                                                  self.opt.divided_number_each_iteration,
                                                                  self.model_dict,
                                                                  intrinsic_matrix=self.cam_K)

        if debug:
            time_used = time.time() - now
            print('zebra time:', time_used)
            
        if success:
            return r_predict, t_predict, True
        else:
            return None, None, False

    def preprocess(self, img, detected_box):
        Bbox = padding_Bbox(detected_box, padding_ratio=self.opt.padding_ratio)
        roi_x = get_roi(img, Bbox, self.opt.bbox_cropsize_img, interpolation=cv2.INTER_NEAREST, resize_method=self.opt.resize_method)
        Bbox = get_final_Bbox(Bbox, self.opt.resize_method, img.shape[1], img.shape[0])
        roi_x = transform_pre(roi_x)
        return roi_x.unsqueeze(0), Bbox
        
        
        
