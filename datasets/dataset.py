import pickle
import numpy as np
import torch
import cv2
from PIL import Image

from torch.utils.data import Dataset
import torchvision.transforms as transforms
from .GDR_Net_Augmentation import get_affine_transform, build_augmentations

class zebra_dataset(Dataset):
    def __init__(self, opt):
        #save options
        if opt.is_train:
            with open('./checkpoints'+opt.save_dir+'/options.pickle', 'wb') as f:
                pickle.dump(opt, f)

        self.opt = opt
        self.folder = opt.dataset_folder
        self.read_raw_data(opt.txt_file_path)

    def read_raw_data(self, file_path):
        self.scene_list = []
        self.frame_list = []
        self.index_list = []
        
        with open(file_path,'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.strip()
                line = line.split()
                self.scene_list.append(line[0])
                self.frame_list.append(line[1])
                self.index_list.append(int(line[2]))

            self.nSamples = len(self.scene_list)

    def __len__(self):
        return self.nSamples

    def __getitem__(self, index):
        rgb_path = self.folder + 'scenes/'+self.scene_list[index]+'/%s.jpg'%self.frame_list[index]
        mask_path = self.folder + 'mask/'+self.scene_list[index]+'/%s.png'%self.frame_list[index]
        pose_path = self.folder + 'pose/'+self.scene_list[index]+'/%s.pickle'%self.frame_list[index]
        bbox_path = self.folder + 'bbox/'+self.scene_list[index]+'/%s.pickle'%self.frame_list[index]
        binary_code_path = self.folder + 'binary_codes/'+self.scene_list[index]+'/%s.pickle'%self.frame_list[index]
        
        full_rgb = cv2.imread(rgb_path)
        full_mask = cv2.imread(mask_path)[:,:,0]
        obj_mask = (full_mask == self.index_list[index]+1).astype(int)

        with open(pose_path, 'rb') as f:
            poses = pickle.load(f)

        pose = poses[self.index_list[index]]
        R_gt = np.array(pose['cam_R_m2c']).reshape(3,3)
        t_gt = np.array(pose['cam_t_m2c']).reshape(3,1)
        
        with open(bbox_path, 'rb') as f:
            bboxes = pickle.load(f)

        Bbox = np.array(bboxes[self.index_list[index]][0])

        with open(binary_code_path, 'rb') as f:
            GT_img = pickle.load(f)

        if self.opt.is_train:
            aug_rgb = self.apply_augmentation(full_rgb)
            
            Bbox = aug_Bbox(Bbox, padding_ratio=self.opt.padding_ratio)
            #Bbox = padding_Bbox(Bbox, padding_ratio=self.opt.padding_ratio)
            
            roi_rgb = get_roi(aug_rgb, Bbox, self.opt.bbox_cropsize_img, interpolation=cv2.INTER_LINEAR, resize_method=self.opt.resize_method)
            roi_GT_img = get_roi(GT_img, Bbox, self.opt.bbox_cropsize_gt, interpolation=cv2.INTER_NEAREST, resize_method=self.opt.resize_method)
            roi_obj_mask = get_roi(obj_mask, Bbox, self.opt.bbox_cropsize_gt, interpolation=cv2.INTER_NEAREST, resize_method=self.opt.resize_method)
            roi_full_mask = get_roi(full_mask, Bbox, self.opt.bbox_cropsize_gt, interpolation=cv2.INTER_NEAREST, resize_method=self.opt.resize_method)

            Bbox = get_final_Bbox(Bbox, self.opt.resize_method, full_rgb.shape[1], full_rgb.shape[0])
            
        else:
            Bbox = padding_Bbox(Bbox, padding_ratio=self.opt.padding_ratio)
            roi_rgb = get_roi(full_rgb, Bbox, self.opt.bbox_cropsize_img, interpolation=cv2.INTER_LINEAR, resize_method=self.opt.resize_method)
            roi_GT_img = get_roi(GT_img, Bbox, self.opt.bbox_cropsize_gt, interpolation=cv2.INTER_NEAREST, resize_method=self.opt.resize_method)
            roi_obj_mask = get_roi(obj_mask, Bbox, self.opt.bbox_cropsize_gt, interpolation=cv2.INTER_NEAREST, resize_method=self.opt.resize_method)
            roi_full_mask = get_roi(full_mask, Bbox, self.opt.bbox_cropsize_gt, interpolation=cv2.INTER_NEAREST, resize_method=self.opt.resize_method)

            Bbox = get_final_Bbox(Bbox, self.opt.resize_method, full_rgb.shape[1], full_rgb.shape[0])
            
        #cv2.imwrite('./visualization/rgb.jpg', roi_rgb)
        #cv2.imwrite('./visualization/mask.jpg',roi_obj_mask*255)

        roi_rgb, roi_full_mask, roi_obj_mask, class_code_images = self.transform_pre(roi_rgb, roi_full_mask, roi_obj_mask, roi_GT_img)

        data = {}
        data['roi_rgb'] = roi_rgb
        data['roi_full_mask'] = roi_full_mask
        data['roi_obj_mask'] = roi_obj_mask
        data['R'] = R_gt
        data['t'] = t_gt
        data['Bbox'] = Bbox
        data['class_code_img'] = class_code_images
        data['cam_K'] = np.array([[616.580322265625,0,323.10302734375],
                          [0,616.7783203125,238.46360778808594],
                          [0,0,1]])
        return data

    def transform_pre(self, sample_x, sample_entire_mask, sample_mask, gt_code):
        composed_transforms_img = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])

        x_pil = Image.fromarray(np.uint8(sample_x)).convert('RGB')

        sample_entire_mask = sample_entire_mask / 255.
        sample_entire_mask = torch.from_numpy(sample_entire_mask).type(torch.float)
        sample_mask = torch.from_numpy(sample_mask).type(torch.float)
        gt_code = torch.from_numpy(gt_code).permute(2, 0, 1) 
    
        return composed_transforms_img(x_pil), sample_entire_mask, sample_mask, gt_code
    
    def apply_augmentation(self, x):
        augmentations = build_augmentations(self.opt.use_peper_salt, self.opt.use_motion_blur)      
        color_aug_prob = 0.8
        if np.random.rand() < color_aug_prob:
            x = augmentations.augment_image(x)
        return x

def aug_Bbox(GT_Bbox, padding_ratio):
    x1 = GT_Bbox[0].copy()
    x2 = GT_Bbox[0] + GT_Bbox[2]
    y1 = GT_Bbox[1].copy()
    y2 = GT_Bbox[1] + GT_Bbox[3]

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1

    scale_ratio = 1 + 0.25 * (2 * np.random.random_sample() - 1)  # [1-0.25, 1+0.25]
    shift_ratio = 0.25 * (2 * np.random.random_sample(2) - 1)  # [-0.25, 0.25]
    bbox_center = np.array([cx + bw * shift_ratio[0], cy + bh * shift_ratio[1]])  # (h/2, w/2)
    # 1.5 is the additional pad scale
    augmented_bw = int(bw * scale_ratio * padding_ratio)
    augmented_bh = int(bh * scale_ratio * padding_ratio)
    
    augmented_Box = np.array([int(bbox_center[0]-augmented_bw/2), int(bbox_center[1]-augmented_bh/2), augmented_bw, augmented_bh])
    return augmented_Box

def padding_Bbox(Bbox, padding_ratio):
    x1 = Bbox[0]
    x2 = Bbox[0] + Bbox[2]
    y1 = Bbox[1]
    y2 = Bbox[1] + Bbox[3]

    cx = 0.5 * (x1 + x2)
    cy = 0.5 * (y1 + y2)
    bh = y2 - y1
    bw = x2 - x1

    padded_bw = int(bw * padding_ratio)
    padded_bh = int(bh * padding_ratio)
        
    padded_Box = np.array([int(cx-padded_bw/2), int(cy-padded_bh/2), int(padded_bw), int(padded_bh)])
    return padded_Box

def get_roi(input, Bbox, crop_size, interpolation, resize_method):
    if resize_method == "crop_resize":
        roi = crop_resize(input, Bbox, crop_size, interpolation = interpolation)
        return roi
    elif resize_method == "crop_resize_by_warp_affine":
        scale, bbox_center = get_scale_and_Bbox_center(Bbox, input)
        roi = crop_resize_by_warp_affine(input, bbox_center, scale, crop_size, interpolation = interpolation)
        return roi
    elif resize_method == "crop_square_resize":
        roi = crop_square_resize(input, Bbox, crop_size, interpolation=interpolation)
        return roi
    else:
        raise NotImplementedError(f"unknown decoder type: {resize_method}")


def get_final_Bbox(Bbox, resize_method, max_x, max_y):
    x1 = Bbox[0]
    bw = Bbox[2]
    x2 = Bbox[0] + bw
    y1 = Bbox[1]
    bh = Bbox[3]
    y2 = Bbox[1] + bh
    if resize_method == "crop_square_resize" or resize_method == "crop_resize_by_warp_affine":
        bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
        if bh > bw:
            x1 = bbox_center[0] - bh/2
            x2 = bbox_center[0] + bh/2
        else:
            y1 = bbox_center[1] - bw/2
            y2 = bbox_center[1] + bw/2
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        Bbox = np.array([x1, y1, x2-x1, y2-y1])

    elif resize_method == "crop_resize":
        x1 = max(x1, 0)
        y1 = max(y1, 0)
        x2 = min(x2, max_x)
        y2 = min(y2, max_y)
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        Bbox = np.array([x1, y1, x2-x1, y2-y1])

    return Bbox

def crop_resize_by_warp_affine(img, center, scale, output_size, rot=0, interpolation=cv2.INTER_LINEAR):
    """
    output_size: int or (w, h)
    NOTE: if img is (h,w,1), the output will be (h,w)
    """
    if isinstance(scale, (int, float)):
        scale = (scale, scale)
    if isinstance(output_size, int):
        output_size = (output_size, output_size)
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img, trans, (int(output_size[0]), int(output_size[1])), flags=interpolation)

    return dst_img

def crop_square_resize(img, Bbox, crop_size=None, interpolation=None):
    x1 = Bbox[0]
    bw = Bbox[2]
    x2 = Bbox[0] + bw
    y1 = Bbox[1]
    bh = Bbox[3]
    y2 = Bbox[1] + bh

    bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
    if bh > bw:
        x1 = bbox_center[0] - bh/2
        x2 = bbox_center[0] + bh/2
    else:
        y1 = bbox_center[1] - bw/2
        y2 = bbox_center[1] + bw/2

    x1 = int(x1)
    y1 = int(y1)
    x2 = int(x2)
    y2 = int(y2)

    if img.ndim > 2:
        roi_img = np.zeros((max(bh, bw), max(bh, bw), img.shape[2]), dtype=img.dtype)
    else:
        roi_img = np.zeros((max(bh, bw), max(bh, bw)), dtype=img.dtype)
    roi_x1 = max((0-x1), 0)
    x1 = max(x1, 0)
    roi_x2 = roi_x1 + min((img.shape[1]-x1), (x2-x1))
    roi_y1 = max((0-y1), 0)
    y1 = max(y1, 0)
    roi_y2 = roi_y1 + min((img.shape[0]-y1), (y2-y1))
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])

    roi_img[roi_y1:roi_y2, roi_x1:roi_x2] = img[y1:y2, x1:x2].copy()
    roi_img = cv2.resize(roi_img, (crop_size,crop_size), interpolation=interpolation)
    return roi_img

def crop_resize(img, Bbox, crop_size=None, interpolation=None):
    x1 = max(0, Bbox[0])
    x2 = min(img.shape[1], Bbox[0]+Bbox[2])
    y1 = max(0, Bbox[1])
    y2 = min(img.shape[0], Bbox[1]+Bbox[3])
    ####
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, img.shape[1])
    y2 = min(y2, img.shape[0])
    ####

    img = img[y1:y2, x1:x2]
    roi_img = cv2.resize(img, (crop_size, crop_size), interpolation = interpolation)
    return roi_img

def get_scale_and_Bbox_center(Bbox, image):
    x1 = Bbox[0]
    bw = Bbox[2]
    x2 = Bbox[0] + bw
    y1 = Bbox[1]
    bh = Bbox[3]
    y2 = Bbox[1] + bh

    bbox_center = np.array([0.5 * (x1 + x2), 0.5 * (y1 + y2)])
    if bh > bw:
        x1 = bbox_center[0] - bh/2
        x2 = bbox_center[0] + bh/2
    else:
        y1 = bbox_center[1] - bw/2
        y2 = bbox_center[1] + bw/2

    scale = max(bh, bw)
    scale = min(scale, max(image.shape[0], image.shape[1])) *1.0
    return scale, bbox_center
        
