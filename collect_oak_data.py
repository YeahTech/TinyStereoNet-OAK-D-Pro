#!/usr/bin/env python3

import cv2
import os
import numpy as np
import depthai as dai
import argparse
import time, datetime


parser = argparse.ArgumentParser()
parser.add_argument(
    "-res",
    "--resolution",
    type=str,
    default="400",
    help="Sets the resolution on mono cameras. Options: 800 | 720 | 400",
)
parser.add_argument(
    "-md",
    "--mesh_dir",
    type=str,
    default=None,
    help="Output directory for mesh files. If not specified mesh files won't be saved",
)
parser.add_argument(
    "-lm",
    "--load_mesh",
    default=False,
    action="store_true",
    help="Read camera intrinsics, generate mesh files and load them into the stereo node.",
)
parser.add_argument(
    "-rect",
    "--out_rectified",
    default=True,
    action="store_true",
    help="Generate and display rectified streams",
)
parser.add_argument(
    "-lr",
    "--lrcheck",
    default=False,
    action="store_true",
    help="Better handling for occlusions",
)
parser.add_argument(
    "-e",
    "--extended",
    default=False,
    action="store_true",
    help="Closer-in minimum depth, disparity range is doubled",
)
parser.add_argument(
    "-s",
    "--subpixel",
    default=True,
    action="store_true",
    help="Better accuracy for longer distance, fractional disparity 32-levels",
)
parser.add_argument(
    "-m",
    "--median",
    type=str,
    default="7x7",
    help="Choose the size of median filtering. Options: OFF | 3x3 | 5x5 | 7x7 (default)",
)
parser.add_argument(
    "-d",
    "--depth",
    default=True,
    action="store_true",
    help="Display depth frames",
)
parser.add_argument(
    "-c",
    "--collect_data",
    default=True,
    action="store_true",
    help="Display depth frames",
)
args = parser.parse_args()

resolutionMap = {"800": (1280, 800), "720": (1280, 720), "400": (640, 400)}
if args.resolution not in resolutionMap:
    exit("Unsupported resolution!")

resolution = resolutionMap[args.resolution]
meshDirectory = args.mesh_dir  # Output dir for mesh files
generateMesh = args.load_mesh  # Load mesh files

outRectified = args.out_rectified  # Output and display rectified streams
lrcheck = args.lrcheck  # Better handling for occlusions
extended = args.extended  # Closer-in minimum depth, disparity range is doubled
subpixel = args.subpixel  # Better accuracy for longer distance, fractional disparity 32-levels
depth = args.depth  # Display depth frames
collect_data = args.collect_data # Collect data

medianMap = {
    "OFF": dai.StereoDepthProperties.MedianFilter.MEDIAN_OFF,
    "3x3": dai.StereoDepthProperties.MedianFilter.KERNEL_3x3,
    "5x5": dai.StereoDepthProperties.MedianFilter.KERNEL_5x5,
    "7x7": dai.StereoDepthProperties.MedianFilter.KERNEL_7x7,
}
if args.median not in medianMap:
    exit("Unsupported median size!")

median = medianMap[args.median]

print("StereoDepth config options:")
print("    Resolution:  ", resolution)
print("    Left-Right check:  ", lrcheck)
print("    Extended disparity:", extended)
print("    Subpixel:          ", subpixel)
print("    Median filtering:  ", median)
print("    Generating mesh files:  ", generateMesh)
print("    Outputting mesh files to:  ", meshDirectory)


def getMesh(calibData):
    M1 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.LEFT, resolution[0], resolution[1]))
    print('Left mono camera Intrinsics:', M1)
    print('Left mono camera focal length in pixels:', M1[0][0])
    d1 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.LEFT))
    R1 = np.array(calibData.getStereoLeftRectificationRotation())
    M2 = np.array(calibData.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT, resolution[0], resolution[1]))
    print('Right mono camera Intrinsics:', M2)
    print('Right mono camera focal length in pixels:', M2[0][0])
    print('Baseline_cm: ', calibData.getBaselineDistance())
    d2 = np.array(calibData.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT))
    R2 = np.array(calibData.getStereoRightRectificationRotation())
    mapXL, mapYL = cv2.initUndistortRectifyMap(M1, d1, R1, M2, resolution, cv2.CV_32FC1)
    mapXR, mapYR = cv2.initUndistortRectifyMap(M2, d2, R2, M2, resolution, cv2.CV_32FC1)

    meshCellSize = 16
    meshLeft = []
    meshRight = []

    for y in range(mapXL.shape[0] + 1):
        if y % meshCellSize == 0:
            rowLeft = []
            rowRight = []
            for x in range(mapXL.shape[1] + 1):
                if x % meshCellSize == 0:
                    if y == mapXL.shape[0] and x == mapXL.shape[1]:
                        rowLeft.append(mapYL[y - 1, x - 1])
                        rowLeft.append(mapXL[y - 1, x - 1])
                        rowRight.append(mapYR[y - 1, x - 1])
                        rowRight.append(mapXR[y - 1, x - 1])
                    elif y == mapXL.shape[0]:
                        rowLeft.append(mapYL[y - 1, x])
                        rowLeft.append(mapXL[y - 1, x])
                        rowRight.append(mapYR[y - 1, x])
                        rowRight.append(mapXR[y - 1, x])
                    elif x == mapXL.shape[1]:
                        rowLeft.append(mapYL[y, x - 1])
                        rowLeft.append(mapXL[y, x - 1])
                        rowRight.append(mapYR[y, x - 1])
                        rowRight.append(mapXR[y, x - 1])
                    else:
                        rowLeft.append(mapYL[y, x])
                        rowLeft.append(mapXL[y, x])
                        rowRight.append(mapYR[y, x])
                        rowRight.append(mapXR[y, x])
            if (mapXL.shape[1] % meshCellSize) % 2 != 0:
                rowLeft.append(0)
                rowLeft.append(0)
                rowRight.append(0)
                rowRight.append(0)

            meshLeft.append(rowLeft)
            meshRight.append(rowRight)

    meshLeft = np.array(meshLeft)
    meshRight = np.array(meshRight)

    return meshLeft, meshRight


def saveMeshFiles(meshLeft, meshRight, outputPath):
    print("Saving mesh to:", outputPath)
    meshLeft.tofile(outputPath + "/left_mesh.calib")
    meshRight.tofile(outputPath + "/right_mesh.calib")


def getDisparityFrame(frame):
    maxDisp = stereo.initialConfig.getMaxDisparity()
    disp = (frame * (255.0 / maxDisp)).astype(np.uint8)
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

    return disp


print("Creating Stereo Depth pipeline")
pipeline = dai.Pipeline()

camLeft = pipeline.create(dai.node.MonoCamera)
camRight = pipeline.create(dai.node.MonoCamera)

stereo = pipeline.create(dai.node.StereoDepth)
xoutLeft = pipeline.create(dai.node.XLinkOut)
xoutRight = pipeline.create(dai.node.XLinkOut)
xoutDisparity = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)
xoutRectifLeft = pipeline.create(dai.node.XLinkOut)
xoutRectifRight = pipeline.create(dai.node.XLinkOut)

camLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
camRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
res = (
    dai.MonoCameraProperties.SensorResolution.THE_800_P
    if resolution[1] == 800
    else dai.MonoCameraProperties.SensorResolution.THE_720_P
    if resolution[1] == 720
    else dai.MonoCameraProperties.SensorResolution.THE_400_P
)
for monoCam in (camLeft, camRight):  # Common config
    monoCam.setResolution(res)
    monoCam.setFps(20.0)

stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_ACCURACY)
stereo.initialConfig.setMedianFilter(median)  # KERNEL_7x7 default
stereo.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
stereo.setLeftRightCheck(lrcheck)
stereo.setExtendedDisparity(extended)
stereo.setSubpixel(subpixel)

xoutLeft.setStreamName("left")
xoutRight.setStreamName("right")
xoutDisparity.setStreamName("disparity")
xoutDepth.setStreamName("depth")
xoutRectifLeft.setStreamName("rectifiedLeft")
xoutRectifRight.setStreamName("rectifiedRight")

camLeft.out.link(stereo.left)
camRight.out.link(stereo.right)
stereo.syncedLeft.link(xoutLeft.input)
stereo.syncedRight.link(xoutRight.input)
stereo.disparity.link(xoutDisparity.input)
if depth:
    stereo.depth.link(xoutDepth.input)
if outRectified:
    stereo.rectifiedLeft.link(xoutRectifLeft.input)
    stereo.rectifiedRight.link(xoutRectifRight.input)

streams = ["left", "right"]
if outRectified:
    streams.extend(["rectifiedLeft", "rectifiedRight"])
streams.append("disparity")
if depth:
    streams.append("depth")

device = dai.Device()
calibData = device.readCalibration()
leftMesh, rightMesh = getMesh(calibData)
if generateMesh:
    meshLeft = list(leftMesh.tobytes())
    meshRight = list(rightMesh.tobytes())
    stereo.loadMeshData(meshLeft, meshRight)

if meshDirectory is not None:
    saveMeshFiles(leftMesh, rightMesh, meshDirectory)

print("Creating DepthAI device")
with device:
    device.startPipeline(pipeline)

    # Create a receive queue for each stream
    qList = [device.getOutputQueue(stream, 8, blocking=False) for stream in streams]

    left_frame_rec = right_frame_rec = None
    save_dir = f'collected_data/{time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())}'
    while True:
        for q in qList:
            name = q.getName()
            output = q.get()
            frame = output.getCvFrame()
            if name == "depth":
                frame_depth = frame.astype(np.uint16)
            elif name == "disparity":
                frame_disp_viz = getDisparityFrame(frame)
                cv2.imshow(name, frame_disp_viz)
            elif name == "rectifiedLeft":
                left_frame_rec = frame
            elif name == "rectifiedRight":
                right_frame_rec = frame
            
            if collect_data:
                if name in ['left', 'right', 'rectifiedLeft', 'rectifiedRight', 'disparity']:
                    today_start = datetime.datetime.today().replace(hour=0, minute=0, second=0, microsecond=0)
                    time_stamp = (today_start + output.getTimestamp()).timestamp()
                    save_name = os.path.join(save_dir, name, f"{time_stamp:016f}.png")
                    save_sub_dir = os.path.dirname(save_name)
                    os.makedirs(save_sub_dir, exist_ok=True)
                    cv2.imwrite(save_name, frame)

            if left_frame_rec is not None and right_frame_rec is not None:
                stereo_image = cv2.hconcat([left_frame_rec, right_frame_rec])
                for i in range(50, stereo_image.shape[0], 50):
                    stereo_image[i, :] = 255
                cv2.imshow('stereo_rectified', stereo_image)
        if cv2.waitKey(1) == ord("q"):
            break