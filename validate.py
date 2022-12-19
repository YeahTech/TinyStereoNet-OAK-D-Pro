import copy
import cv2
import os
import sys
import logging
import glob

import numpy as np
from tqdm import tqdm
from tqdm.contrib import tzip


MIN_DISP = 0.1
MAX_DISP = 256


def visualize_disp(disp, color=cv2.COLORMAP_INFERNO):
    disp_vis = (disp - disp.min()) / (disp.max() - disp.min()) * 255.0
    disp_vis = disp_vis.astype("uint8")
    disp_vis = cv2.applyColorMap(disp_vis, color)
    return disp_vis


def cal_pix_error(pred, gt, max_diff=3, start=None, end=None, step=None, border=0):
    img_value = []
    if border > 0:
        pred = pred[border:-border, border:-border]
        gt = gt[border:-border, border:-border]
    mask = (gt > MIN_DISP) & (gt < MAX_DISP) & (
        pred > 0)  # just stat in the mask
    gt[~mask] = 0
    pred[~mask] = 0
    if np.sum(mask) > 0:
        value = abs(pred[mask] - gt[mask]) > max_diff
        img_value.append((value.sum(), np.sum(mask)))
    else:
        img_value.append((0, 0))
    if start is not None and end is not None and step is not None:
        for i in range(start, end, step):
            mask = (i < gt) & (gt <= i + step)
            if np.sum(mask) > 0:
                value = abs(pred[mask] - gt[mask]) > max_diff
                img_value.append((value.sum(), np.sum(mask)))
            else:
                img_value.append((0, 0))
    return img_value


def draw_error_maps(left_img, pred, gt):
    pred_gt_viz = visualize_disp(cv2.hconcat([pred, gt]))
    diff = np.abs(pred-gt)
    diff_viz = visualize_disp(diff, color=cv2.COLORMAP_JET)
    show = cv2.hconcat([left_img, pred_gt_viz, diff_viz])
    error_maps = []

    for thresh in [3.0, 2.0, 1.0, 0.5]:
        error_map = ((diff > thresh)*255).astype(np.uint8)
        error_maps.append(error_map)
    error = cv2.hconcat(error_maps)
    error = cv2.cvtColor(error, cv2.COLOR_GRAY2BGR)
    show = cv2.vconcat([show, error])
    return show


def cal_mask_epe(pred, gt, start=None, end=None, step=None, border=0):
    img_value = []
    if border > 0:
        pred = pred[border:-border, border:-border]
        gt = gt[border:-border, border:-border]
    mask = (gt > MIN_DISP) & (gt < MAX_DISP) & (
        pred > 0)  # just stat in the mask
    gt[~mask] = 0
    pred[~mask] = 0
    if np.sum(mask) > 0:
        value = abs(pred[mask] - gt[mask])
        img_value.append((value.sum(), np.sum(mask)))
    else:
        img_value.append((0, 0))
    if start is not None and end is not None and step is not None:
        for i in range(start, end, step):
            mask = (i < gt) & (gt <= i + step)
            if np.sum(mask) > 0:
                value = abs(pred[mask] - gt[mask])
                img_value.append((value.sum(), np.sum(mask)))
            else:
                img_value.append((0, 0))
    return img_value


def eval_disparity(left_paths, disp_pred_paths, disp_gt_paths, pred_loader=None, gt_loader=None, max_disp=95, save_result='result'):
    pix_1 = []
    pix_2 = []
    pix_3 = []
    epe = []

    os.makedirs(save_result, exist_ok=True)
    eval_dict = {}
    for left_path, disp_pred_path, disp_gt_path in zip(left_paths, disp_pred_paths, disp_gt_paths):
        left_img = cv2.imread(left_path)
        if pred_loader is None:
            pred = cv2.imread(disp_pred_path, cv2.IMREAD_UNCHANGED)
        else:
            pred = pred_loader(disp_pred_path)

        if gt_loader is None:
            g = cv2.imread(disp_gt_path, cv2.IMREAD_UNCHANGED)
        else:
            g = gt_loader(disp_gt_path)

        print(g[89, 408])
        print(pred[89, 408])

        m = (g > 0) & (g < max_disp) & (left_img[:, :, 0] != 0)
        g[~m] = 0
        pix_1_img = cal_pix_error(pred, g, 1)
        pix_2_img = cal_pix_error(pred, g, 2)
        pix_3_img = cal_pix_error(pred, g, 3)
        epe_img = cal_mask_epe(pred, g)
        pix_1.append(pix_1_img)
        pix_2.append(pix_2_img)
        pix_3.append(pix_3_img)
        epe.append(epe_img)
        info = f'{left_path} pix_1={pix_1_img[0][0]/pix_1_img[0][1]*100:.2f}% pix_2={pix_2_img[0][0]/pix_2_img[0][1]*100:.2f}% pix_3={pix_3_img[0][0]/pix_3_img[0][1]*100:.2f}% epe={epe_img[0][0]/epe_img[0][1]:.2f}'
        print(info)
        eval_dict[left_path] = {'pix_1': pix_1_img[0][0]/pix_1_img[0][1]*100,
                                'pix_2': pix_2_img[0][0]/pix_2_img[0][1]*100,
                                'pix_3': pix_3_img[0][0]/pix_3_img[0][1]*100,
                                'epe': epe_img[0][0]/epe_img[0][1],
                                }

        # draw
        show = draw_error_maps(left_img, pred, g)
        save_name = os.path.join(save_result, os.path.basename(left_path))
        cv2.imwrite(save_name, show)

    pix_1 = np.array(pix_1)
    pix_2 = np.array(pix_2)
    pix_3 = np.array(pix_3)
    epe = np.array(epe)
    pix_1 = np.sum(pix_1[:, 0, 0]) / (np.sum(pix_1[:, 0, 1]) + 1e-7)
    pix_2 = np.sum(pix_2[:, 0, 0]) / (np.sum(pix_2[:, 0, 1]) + 1e-7)
    pix_3 = np.sum(pix_3[:, 0, 0]) / (np.sum(pix_3[:, 0, 1]) + 1e-7)
    epe = np.sum(epe[:, 0, 0]) / (np.sum(epe[:, 0, 1]) + 1e-7)

    infos = []
    infos.append(f'pix_1={pix_1:.2%}')
    infos.append(f'pix_2={pix_2:.2%}')
    infos.append(f'pix_3={pix_3:.2%}')
    infos.append(f'epe={epe:.2}')
    infos = ', '.join(infos)

    # item = sorted(eval_dict.items(), key = lambda kv:kv[1]['pix_1'])
    # print(item)

    logging.warning(infos)


def oak_disp_png_loader(disp_path):
    disp_u16 = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
    disp_fp32 = disp_u16.astype(np.float32)
    disp_fp32 = disp_fp32 / 8.0
    return disp_fp32


def disp_tiff_loader(disp_path):
    disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
    return disp


if __name__ == '__main__':
    val_list_f = sys.argv[1]
    left_paths = []
    right_paths = []
    disp_paths = []
    with open(val_list_f, 'r') as fin:
        for line in fin:
            left_path, right_path, disp_path = line.strip().split(' ')
            print(left_path)
            left_paths.append(left_path)
            right_paths.append(right_path)
            disp_paths.append(disp_path)

    disp_gt_paths = []
    for left_path in left_paths:
        disp_gt_paths.append(left_path.replace(
            'rectifiedLeft', 'CREStereo').replace('png', 'tiff'))

    eval_disparity(left_paths, disp_paths, disp_gt_paths,
                   oak_disp_png_loader, disp_tiff_loader)
