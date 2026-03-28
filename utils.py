import numpy as np
import torch
from torch.utils.data.distributed import DistributedSampler
import cv2
import os
import pickle
import json
from binary_code_helper.CNN_output_to_pose import load_dict_class_id_3D_points, CNN_outputs_to_object_pose
import sys
sys.path.append('../bop_toolkit')
from bop_toolkit_lib import pose_error, inout
from tqdm import tqdm
from tools_for_BOP import bop_io
from tools_for_BOP.common_dataset_info import get_obj_info
from datasets.bop_dataset_pytorch import bop_dataset_single_obj_pytorch
from datasets.bop_dataset_pytorch_test_dataset import bop_dataset_single_obj_pytorch_dataset

def evaluate_model(model, zebra_net, model_info, dataloader, writer, opt, step, device, ignore_n_bit=0, calc_add_and_adi=True):
    obj_id, obj_diameter, symmetry, dict_class_id_3D_points, vertices = model_info
    
    if symmetry:
        Calculate_Pose_Error_Main = Calculate_ADI_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADD_Error_BOP
        main_metric_name='ADI'
        supp_metric_name='ADD'
    else:
        Calculate_Pose_Error_Main = Calculate_ADD_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADI_Error_BOP
        main_metric_name = 'ADD'
        supp_metric_name = 'ADI'

    if model.binarycode_loss.histogram is not None:
        np.set_printoptions(formatter={'float': lambda x: "{0:.2f}".format(x)})
        print('Train err:{}'.format(model.binarycode_loss.histogram.detach().cpu().numpy()))
    
    zebra_net.eval()
    ADX_passed=np.zeros(len(dataloader.dataset))
    ADX_error=np.zeros(len(dataloader.dataset))
    AUC_ADX_error=np.zeros(len(dataloader.dataset))
    if calc_add_and_adi:
        ADY_passed=np.zeros(len(dataloader.dataset))
        ADY_error=np.zeros(len(dataloader.dataset))
        AUC_ADY_error=np.zeros(len(dataloader.dataset))

    for batch_idx, (data, entire_masks, obj_masks, Rs, ts, Bboxes, class_code_images, cam_Ks) in enumerate(tqdm(dataloader)):
        inputs = data.to(device)

        with torch.no_grad():
            pred_masks_prob, pred_code_prob = zebra_net(inputs)
        pred_codes = from_output_to_class_binary_code(pred_code_prob, opt.BinaryCode_Loss_Type,
                                                      divided_num_each_iteration=opt.divided_number_each_iteration,
                                                      binary_code_length=opt.binary_code_length)
        pred_masks = from_output_to_class_mask(pred_masks_prob)
        pred_codes = pred_codes.transpose(0,2,3,1)
        pred_masks = pred_masks.transpose(0,2,3,1)
        pred_masks = pred_masks.squeeze(axis=-1)

        Rs = Rs.numpy()
        ts = ts.numpy()
        Bboxes = Bboxes.numpy()
        cam_Ks = cam_Ks.numpy()

        for counter, (r_GT, t_GT, Bbox, cam_K) in enumerate(zip(Rs, ts, Bboxes, cam_Ks)):
            if ignore_n_bit!=0 and opt.eval_with_ignore_bits:
                R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[counter], pred_codes[counter][:,:,:-ignore_n_bit],
                                                                            Bbox, opt.bbox_cropsize_gt, opt.divided_number_each_iteration, dict_class_id_3D_points, 
                                                                            intrinsic_matrix=cam_K)
            else:
                R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[counter], pred_codes[counter], 
                                                                            Bbox, opt.bbox_cropsize_gt, opt.divided_number_each_iteration, dict_class_id_3D_points, 
                                                                            intrinsic_matrix=cam_K)
        
            batch_size = dataloader.batch_size
            sample_idx = batch_idx*batch_size + counter
            adx_erro = 10000
            if success:
                adx_error = Calculate_Pose_Error_Main(r_GT, t_GT, R_predict, t_predict, vertices)
                if np.isnan(adx_error):
                    adx_error = 10000
            if adx_error < obj_diameter*0.1:
                ADX_passed[sample_idx] = 1
            else:
                pass
                #print(adx_error)
                #print(r_GT)
                #print(R_predict)
                #print(t_GT)
                #print(t_predict)
            ADX_error[sample_idx] = adx_error
            AUC_ADX_error[sample_idx] = min(100,max(100-adx_error,0))
            if calc_add_and_adi:
                ady_error = 10000
                if success:
                    ady_error = Calculate_Pose_Error_Supp(r_GT, t_GT, R_predict, t_predict, vertices)
                    if np.isnan(ady_error):
                        ady_error = 10000
                if ady_error < obj_diameter*0.1:
                    ADY_passed[sample_idx] = 1
                ADY_error[sample_idx] = ady_error
                AUC_ADY_error[sample_idx] = min(100,max(100-ady_error,0))
    
    ADX_passed = np.mean(ADX_passed)
    ADX_error = np.mean(ADX_error)
    AUC_ADX_error = np.mean(AUC_ADX_error)
    writer.add_scalar('TESTDATA_{}/{}_test'.format(main_metric_name,main_metric_name), ADX_passed, step)
    writer.add_scalar('TESTDATA_{}/{}_Error_test'.format(main_metric_name,main_metric_name), ADX_error, step)
    writer.add_scalar('TESTDATA_AUC_{}/AUC_{}_Error_test'.format(main_metric_name,main_metric_name), AUC_ADX_error, step)
    if calc_add_and_adi:
        ADY_passed = np.mean(ADY_passed)
        ADY_error= np.mean(ADY_error)
        AUC_ADY_error = np.mean(AUC_ADY_error)
        writer.add_scalar('TESTDATA_{}/{}_test'.format(supp_metric_name,supp_metric_name), ADY_passed, step)
        writer.add_scalar('TESTDATA_{}/{}_Error_test'.format(supp_metric_name,supp_metric_name), ADY_error, step)
        writer.add_scalar('TESTDATA_AUC_{}/AUC_{}_Error_test'.format(supp_metric_name,supp_metric_name), AUC_ADY_error, step)

    return ADX_passed, ADX_error, AUC_ADX_error


def evaluate_train_model(model, zebra_net, model_info, opt, data, obj_masks, Rs, ts, Bboxes, cam_Ks, ignore_n_bit, device):
    obj_id, obj_diameter, symmetry, dict_class_id_3D_points, vertices = model_info
    
    if symmetry:
        Calculate_Pose_Error_Main = Calculate_ADI_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADD_Error_BOP
    else:
        Calculate_Pose_Error_Main = Calculate_ADD_Error_BOP
        Calculate_Pose_Error_Supp = Calculate_ADI_Error_BOP

    ADX_passed=np.zeros(data.shape[0])
    ADX_error=np.zeros(data.shape[0])
    AUC_ADX_error=np.zeros(data.shape[0])
        
    zebra_net.eval()
    inputs = data.to(device)

    with torch.no_grad():
        pred_masks_prob, pred_code_prob = zebra_net(inputs)
    pred_codes = from_output_to_class_binary_code(pred_code_prob, opt.BinaryCode_Loss_Type,
                                                divided_num_each_iteration=opt.divided_number_each_iteration,
                                                binary_code_length=opt.binary_code_length)

    pred_masks = from_output_to_class_mask(pred_masks_prob)
    pred_codes = pred_codes.transpose(0,2,3,1)
    pred_masks = pred_masks.transpose(0,2,3,1)
    pred_masks = pred_masks.squeeze(axis=-1)

    Rs = Rs.numpy()
    ts = ts.numpy()
    Bboxes = Bboxes.numpy()
    cam_Ks = cam_Ks.numpy()

    for counter, (r_GT, t_GT, Bbox, cam_K) in enumerate(zip(Rs, ts, Bboxes, cam_Ks)):
        if ignore_n_bit!=0 and opt.eval_with_ignore_bits:
            R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[counter], pred_codes[counter][:,:,:-ignore_n_bit],
                                                                        Bbox, opt.bbox_cropsize_gt, opt.divided_number_each_iteration, dict_class_id_3D_points, 
                                                                        intrinsic_matrix=cam_K)
        else:
            R_predict, t_predict, success = CNN_outputs_to_object_pose(pred_masks[counter], pred_codes[counter], 
                                                                        Bbox, opt.bbox_cropsize_gt, opt.divided_number_each_iteration, dict_class_id_3D_points, 
                                                                        intrinsic_matrix=cam_K)
        adx_error = 10000
        if success:
            adx_error = Calculate_Pose_Error_Main(r_GT, t_GT, R_predict, t_predict, vertices)
            if np.isnan(adx_error):
                adx_error = 10000
        if adx_error < obj_diameter*0.1:
            ADX_passed[counter] = 1
        ADX_error[counter] = adx_error
        AUC_ADX_error[counter] = min(100,max(100-adx_error,0))

    ADX_passed = np.mean(ADX_passed)
    ADX_error = np.mean(ADX_error)
    AUC_ADX_error = np.mean(AUC_ADX_error)
    return ADX_passed, ADX_error, AUC_ADX_error

def from_output_to_class_mask(pred_mask_prob, thershold=0.5):
    activation_function = torch.nn.Sigmoid()
    pred_mask_prob = activation_function(pred_mask_prob)
    pred_mask_prob = pred_mask_prob.detach().cpu().numpy()
    pred_mask = np.zeros(pred_mask_prob.shape)
    pred_mask[pred_mask_prob>thershold] = 1.
    return pred_mask

def from_output_to_class_binary_code(pred_code_prob, BinaryCode_Loss_Type, thershold=0.5, divided_num_each_iteration=2, binary_code_length=16):
    if BinaryCode_Loss_Type == "BCE" or BinaryCode_Loss_Type == "L1":   
        activation_function = torch.nn.Sigmoid()
        pred_code_prob = activation_function(pred_code_prob)
        pred_code_prob = pred_code_prob.detach().cpu().numpy()
        pred_code = np.zeros(pred_code_prob.shape)
        pred_code[pred_code_prob>thershold] = 1.

    elif BinaryCode_Loss_Type == "CE":   
        activation_function = torch.nn.Softmax(dim=1)
        pred_code_prob = pred_code_prob.reshape(-1, divided_num_each_interation, pred_code_prob.shape[2], pred_code_prob.shape[3])
        pred_code_prob = activation_function(pred_code_prob)
        pred_code_prob = pred_code_prob.detach().cpu().numpy()
        pred_code = np.argmax(pred_code_prob, axis=1)
        pred_code = np.expand_dims(pred_code, axis=1)
        pred_code = pred_code.reshape(-1, binary_code_length, pred_code.shape[2], pred_code.shape[3])
        pred_code_prob = pred_code_prob.max(axis=1, keepdims=True)
        pred_code_prob = pred_code_prob.reshape(-1, binary_code_length, pred_code_prob.shape[2], pred_code_prob.shape[3])
    return pred_code

def Calculate_ADD_Error_BOP(R_GT,t_GT, R_predict, t_predict, vertices):
    t_GT = t_GT.reshape((3,1))
    t_predict = np.array(t_predict).reshape((3,1))

    return pose_error.add(R_predict, t_predict, R_GT, t_GT, vertices)

def Calculate_ADI_Error_BOP(R_GT,t_GT, R_predict, t_predict, vertices):
    t_GT = t_GT.reshape((3,1))
    t_predict = np.array(t_predict).reshape((3,1))

    return pose_error.adi(R_predict, t_predict, R_GT, t_GT, vertices)

def save_options(opt, best_ADD):
    opt.best_ADD = best_ADD
    opt_dict = opt.__dict__

    save_path = './checkpoints' + opt.save_dir + '/options.json'
    with open(save_path, 'w') as f:
        json.dump(opt_dict, f, ensure_ascii=False, indent=4)

def get_train_dataloader(opt, data_folder, batch_size):
    dataset_dir, source_dir, model_plys, model_info, model_ids, rgb_files, depth_files, mask_files, mask_visib_files, gts, gt_infos, cam_param_global, cam_params = bop_io.get_dataset(opt.bop_path,
                                                                                                                                                                                       opt.dataset_name,
                                                                                                                                                                                       train=True,
                                                                                                                                                                                       data_folder= data_folder,
                                                                                                                                                                                       data_per_obj=True,
                                                                                                                                                                                       incl_param=True,
                                                                                                                                                                                       train_obj_visible_theshold=opt.train_obj_visible_threshold
                                                                                                                                                                                       )

    obj_name_obj_id, symmetry_obj = get_obj_info(opt.dataset_name)
    obj_id = int(obj_name_obj_id[opt.obj_name] - 1)

    if opt.obj_name in symmetry_obj:
        symmetry = True
    else:
        symmetry = False

    mesh_path = model_plys[obj_id+1]
    obj_diameter = model_info[str(obj_id+1)]['diameter']

    path_dict = os.path.join(dataset_dir, 'models_GT_color', 'Class_CorresPoint{:06d}.txt'.format(obj_id+1))
    total_number_class, _, _, dict_class_id_3D_points = load_dict_class_id_3D_points(path_dict)
    total_number_class = int(total_number_class)
    GT_code_infos = [opt.divided_number_each_iteration, opt.binary_code_length, total_number_class]

    vertices = inout.load_ply(mesh_path)['pts']
    
    train_dataset = bop_dataset_single_obj_pytorch(
        dataset_dir, data_folder, rgb_files[obj_id], mask_files[obj_id], mask_visib_files[obj_id],
        gts[obj_id], gt_infos[obj_id], cam_params[obj_id], True, opt.bbox_cropsize_img, opt.bbox_cropsize_gt,
        GT_code_infos, padding_ratio=opt.padding_ratio, resize_method=opt.resize_method, use_peper_salt=opt.use_peper_salt,
        use_motion_blur=opt.use_motion_blur, sym_aware_training=opt.sym_aware_training
        )

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=opt.num_workers, drop_last=True,
                                               sampler=DistributedSampler(train_dataset))
    return train_loader, (obj_id, obj_diameter, symmetry, dict_class_id_3D_points, vertices)
    

def get_test_dataloader(opt, obj_id):
    dataset_dir_test,_,_,_,_,test_rgb_files,_,test_mask_files,test_mask_visib_files,test_gts,test_gt_infos,_,camera_params_test = bop_io.get_dataset(
        opt.bop_path,
        opt.dataset_name,
        train=False,
        data_folder=opt.val_folder,
        data_per_obj=True,
        incl_param=True,
        train_obj_visible_theshold=opt.train_obj_visible_threshold
        )

##        if dataset_name == 'ycbv':
##            print("select key frames from ycbv test images")
##            key_frame_index = ycbv_select_keyframe(Detection_reaults, test_rgb_files[obj_id])
##            test_rgb_files_keyframe = [test_rgb_files[obj_id][i] for i in key_frame_index]
##            test_mask_files_keyframe = [test_mask_files[obj_id][i] for i in key_frame_index]
##            test_mask_visib_files_keyframe = [test_mask_visib_files[obj_id][i] for i in key_frame_index]
##            test_gts_keyframe = [test_gts[obj_id][i] for i in key_frame_index]
##            test_gt_infos_keyframe = [test_gt_infos[obj_id][i] for i in key_frame_index]
##            camera_params_test_keyframe = [camera_params_test[obj_id][i] for i in key_frame_index]
##            test_rgb_files[obj_id] = test_rgb_files_keyframe
##            test_mask_files[obj_id] = test_mask_files_keyframe
##            test_mask_visib_files[obj_id] = test_mask_visib_files_keyframe
##            test_gts[obj_id] = test_gts_keyframe
##            test_gt_infos[obj_id] = test_gt_infos_keyframe
##            camera_params_test[obj_id] = camera_params_test_keyframe
    total_number_class = opt.divided_number_each_iteration**opt.binary_code_length
    GT_code_infos = [opt.divided_number_each_iteration, opt.binary_code_length, total_number_class]
    
    test_dataset = bop_dataset_single_obj_pytorch(
        dataset_dir_test, opt.val_folder, test_rgb_files[obj_id], test_mask_files[obj_id], test_mask_visib_files[obj_id],
        test_gts[obj_id], test_gt_infos[obj_id], camera_params_test[obj_id], False, opt.bbox_cropsize_img, opt.bbox_cropsize_gt, GT_code_infos,
        padding_ratio=opt.padding_ratio, resize_method=opt.resize_method, use_peper_salt=opt.use_peper_salt, use_motion_blur=opt.use_motion_blur,
        sym_aware_training=opt.sym_aware_training
        )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=opt.num_workers)
    return test_loader

def get_test_dataset_dataloader(opt):
    dataset_dir_test,_,_,_,_,test_rgb_files,_,test_mask_files,test_mask_visib_files,test_gts,test_gt_infos,_,camera_params_test = bop_io.get_dataset(
        opt.bop_path,
        opt.dataset_name,
        train=False,
        data_folder=opt.val_folder,
        data_per_obj=True,
        incl_param=True,
        train_obj_visible_theshold=opt.train_obj_visible_threshold
        )

    obj_name_obj_id, symmetry_obj = get_obj_info(opt.dataset_name)
    obj_id = int(obj_name_obj_id[opt.obj_name] - 1)

    total_number_class = opt.divided_number_each_iteration**opt.binary_code_length
    GT_code_infos = [opt.divided_number_each_iteration, opt.binary_code_length, total_number_class]
    
    test_dataset = bop_dataset_single_obj_pytorch_dataset(
        opt, dataset_dir_test, opt.val_folder, test_rgb_files[obj_id], test_mask_files[obj_id], test_mask_visib_files[obj_id],
        test_gts[obj_id], test_gt_infos[obj_id], camera_params_test[obj_id], False, opt.bbox_cropsize_img, opt.bbox_cropsize_gt, GT_code_infos,
        padding_ratio=opt.padding_ratio, resize_method=opt.resize_method, use_peper_salt=opt.use_peper_salt, use_motion_blur=opt.use_motion_blur,
        sym_aware_training=opt.sym_aware_training
        )

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=opt.num_workers)
    return test_loader, obj_id
