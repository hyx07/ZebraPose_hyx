import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2

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

def transform_pre(sample_x):
    composed_transforms_img = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    x_pil = Image.fromarray(np.uint8(sample_x)).convert('RGB')
    return composed_transforms_img(x_pil)
