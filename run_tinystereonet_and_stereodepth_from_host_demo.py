#!/usr/bin/env python3

import cv2
import os
import depthai as dai
import numpy as np
from depthai_sdk import FPSHandler
import argparse
import time

def vis_uncertainty(uncertainty):
    uncertainty_vis = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min()) * 255.0
    uncertainty_vis = uncertainty_vis.astype("uint8")
    uncertainty_vis = cv2.applyColorMap(uncertainty_vis, cv2.COLORMAP_JET)
    return uncertainty_vis

def vis_disparity(disp, scale):
    disp = disp * scale
    disp = disp.astype("uint8")
    disp_vis = cv2.applyColorMap(disp, cv2.COLORMAP_INFERNO)
    return disp_vis

def cvmat2frame(img, width, height, instanceNum):
    data = cv2.resize(img, (width, height))
    if len(data.shape) == 3 and data.shape[2] == 3:
        data = data.transpose((2, 0, 1))
        type = dai.ImgFrame.Type.BGR888p
    else:
        type = dai.ImgFrame.Type.RAW8
    # data = data.reshape(height*width*3)
    tstamp = time.time()
    img = dai.ImgFrame()
    img.setFrame(data)
    img.setTimestamp(tstamp)
    img.setInstanceNum(instanceNum)
    img.setType(type)
    img.setWidth(width)
    img.setHeight(height)
    return img

parser = argparse.ArgumentParser()
parser.add_argument('-nn', '--nn_path', type=str, help="select model blob path for inference", default='models/tinystereonet_320x640_oak6data_maxdisp64.blob')
parser.add_argument('--stereo_image_list', type=str, help="stereo images list", required=True)
parser.add_argument('-shape', '--shape', type=str, help="model input shape, same as used blob", choices=["320x640"], default="320x640")
parser.add_argument('-target_shape', '--target_shape', type=str, help="finial shape for disparity", choices=["400x640"], default="400x640")
parser.add_argument('-nn_threshold', '--nn_threshold', type=float, help="tinystereonet uncertainty threshold", default=0.8)
parser.add_argument('-stereodepth_threshold', '--stereodepth_threshold', type=float, help="stereodepth confidence threshold", default=0.8)

args = parser.parse_args()

NN_PATH = args.nn_path
TARGET_SHAPE = [int(dim) for dim in args.target_shape.split("x")]
NN_SHAPE = [int(dim) for dim in args.shape.split("x")]
resolution = NN_SHAPE[::-1]

# Create pipeline
pipeline = dai.Pipeline()
pipeline.setOpenVINOVersion(dai.OpenVINO.VERSION_2021_4)

# Define StereoDepth
extended_disparity = False
subpixel = True
lr_check = True
depth = pipeline.create(dai.node.StereoDepth)
depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
depth.initialConfig.setMedianFilter(dai.MedianFilter.KERNEL_7x7)
conf_thresh = int(255 * args.stereodepth_threshold)
depth.initialConfig.setConfidenceThreshold(conf_thresh)
depth.setLeftRightCheck(lr_check)
depth.setExtendedDisparity(extended_disparity)
depth.setSubpixel(subpixel)
depth.setInputResolution(resolution)
depth.setRectification(False)

# Define sources and outputs
monoLeft = pipeline.create(dai.node.XLinkIn)
monoRight = pipeline.create(dai.node.XLinkIn)
monoLeft.setStreamName('in_left')
monoRight.setStreamName('in_right')

# NN
nn = pipeline.createNeuralNetwork()
nn.setBlobPath(NN_PATH)
nn.setNumInferenceThreads(2)

# Set NN left/right inputs
monoLeft.out.link(nn.inputs["imgL"])
monoRight.out.link(nn.inputs["imgR"])

# set depth input
monoLeft.out.link(depth.left)
monoRight.out.link(depth.right)

# NN and stereoDepth outputs
nn_xout = pipeline.createXLinkOut()
nn_xout.setStreamName("nn")
nn.out.link(nn_xout.input)

xoutRectifLeft = pipeline.create(dai.node.XLinkOut)
xoutRectifRight = pipeline.create(dai.node.XLinkOut)
xoutRectifLeft.setStreamName("rectifiedLeft")
xoutRectifRight.setStreamName("rectifiedRight")
monoLeft.out.link(xoutRectifLeft.input)
monoRight.out.link(xoutRectifRight.input)

xout = pipeline.create(dai.node.XLinkOut)
xout.setStreamName("disparity")
depth.disparity.link(xout.input)
xout_confidence = pipeline.create(dai.node.XLinkOut)
xout_confidence.setStreamName("confidenceMap")
depth.confidenceMap.link(xout_confidence.input)

# Connect to device and start pipeline
with dai.Device(pipeline) as device:

    inStreams = ['in_left', 'in_right']
    inStreamsCameraID = [dai.CameraBoardSocket.RIGHT, dai.CameraBoardSocket.LEFT]
    in_q_list = []
    for s in inStreams:
        q = device.getInputQueue(s)
        in_q_list.append(q)

    qNn = device.getOutputQueue(name="nn", maxSize=2, blocking=False)
    qDisp = device.getOutputQueue(name="disparity", maxSize=2, blocking=False)
    qConfidence = device.getOutputQueue(name="confidenceMap", maxSize=2, blocking=False)
    qRectifiedLeft = device.getOutputQueue(name="rectifiedLeft", maxSize=2, blocking=False)
    qRectifiedRight = device.getOutputQueue(name="rectifiedRight", maxSize=2, blocking=False)

    fps_handler = FPSHandler()

    font = cv2.FONT_HERSHEY_SIMPLEX
    fontScale = 0.5
    color = (255,0,255)
    thickness = 1

    model_max_disp = 64
    nn_disp_multiplier =  255.0 / model_max_disp
    scale_multiplier = TARGET_SHAPE[1] / NN_SHAPE[1]

    stereo_paths = []
    with open(args.stereo_image_list, 'r') as fin:
        for line in fin:
            left_p, right_p = line.strip().split(' ')
            stereo_paths.append((left_p, right_p))

    for left_p, right_p in stereo_paths:
        # Handle input streams, if any
        left_img = cv2.imread(left_p)
        left_frame = cvmat2frame(left_img, NN_SHAPE[1], NN_SHAPE[0], inStreamsCameraID[0])

        right_img = cv2.imread(right_p)
        right_frame = cvmat2frame(right_img, NN_SHAPE[1], NN_SHAPE[0], inStreamsCameraID[1])

        in_q_list[0].send(left_frame)
        in_q_list[1].send(right_frame)
        
        fps_handler.tick("nn")
        fps = fps_handler.tickFps("nn")

        nnData = qNn.get() # Blocking
        stereodepth_disp = qDisp.get().getFrame()
        stereodepth_confidence = qConfidence.get().getFrame()
        rectified_left = qRectifiedLeft.get().getCvFrame()
        rectified_right = qRectifiedRight.get().getCvFrame()

        # parser nndata
        nn_disp = np.array(nnData.getLayerFp16("disp")).reshape((320, NN_SHAPE[1]))
        nn_disp = cv2.resize(nn_disp, (TARGET_SHAPE[1], TARGET_SHAPE[0])) * scale_multiplier
        nn_uncertainty = np.array(nnData.getLayerFp16("uncertainty")).astype(np.float32).reshape((NN_SHAPE[0], NN_SHAPE[1]))
        nn_uncertainty = cv2.resize(nn_uncertainty, (TARGET_SHAPE[1], TARGET_SHAPE[0]))
        nn_uncertainty_viz = vis_uncertainty(nn_uncertainty)
        nn_disp[nn_uncertainty > args.nn_threshold] = 0.0
        nn_disp_vis = vis_disparity(nn_disp, nn_disp_multiplier)
        # nn_disp_vis = cv2.putText(nn_disp_vis, f"NN FPS {fps:.2f}", (20, 20), font, fontScale, color, thickness, cv2.LINE_AA)
        nn_mask = np.zeros_like(nn_uncertainty, dtype=np.uint8)
        nn_mask[nn_uncertainty > args.nn_threshold] = 255

        # parser stereodepth data
        stereodepth_disp = stereodepth_disp.astype(np.float32) / 8.0
        stereodepth_disp = cv2.resize(stereodepth_disp, (TARGET_SHAPE[1], TARGET_SHAPE[0]), cv2.INTER_NEAREST)
        stereodepth_confidence = cv2.resize(stereodepth_confidence, (TARGET_SHAPE[1], TARGET_SHAPE[0]))
        stereodepth_disp_viz = vis_disparity(stereodepth_disp, nn_disp_multiplier)
        stereodepth_mask = np.zeros_like(stereodepth_confidence, dtype=np.uint8) 
        stereodepth_mask[stereodepth_confidence > conf_thresh] = 255
        stereodepth_confidence_viz = vis_uncertainty(stereodepth_confidence)

        # show
        stereo_img = cv2.hconcat([rectified_left, rectified_right])
        stereo_img_viz = stereo_img
        disparity_viz = cv2.hconcat([nn_disp_vis, stereodepth_disp_viz])
        uncertainty_viz = cv2.hconcat([nn_uncertainty_viz, stereodepth_confidence_viz])
        filter_mask = cv2.hconcat([nn_mask, stereodepth_mask])
        filter_mask_viz = cv2.cvtColor(filter_mask, cv2.COLOR_GRAY2BGR)

        show = cv2.vconcat([stereo_img_viz, disparity_viz, uncertainty_viz, filter_mask_viz])
        save_dir = 'show_result'
        os.makedirs(save_dir, exist_ok=True)
        save_nn = os.path.join(save_dir, '-'.join(left_p.split('/')[-2:])   + '_nn.tiff')
        cv2.imwrite(save_nn, nn_disp)
        save_stereodepth = os.path.join(save_dir, '-'.join(left_p.split('/')[-2:])   + '_stereodepth.tiff')
        cv2.imwrite(save_stereodepth, stereodepth_disp)
        save_name = os.path.join(save_dir, '-'.join(left_p.split('/')[-2:])   + '_show.jpg')
        print(save_name)
        cv2.imwrite(save_name, show)
        show_downsample = cv2.resize(show, (show.shape[1]//2, show.shape[0]//2))
        cv2.imshow('tinyestereonet vs stereodepth', show_downsample)
        if cv2.waitKey(1) == ord('q'):
            break
