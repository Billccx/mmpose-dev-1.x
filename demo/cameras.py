import pyrealsense2 as rs
import numpy as np
import cv2
import os
# from utils.color_img import ColorImg

class ColorImg:
    def __init__(self,colorframe,rs_intrinsics):
        self.colorframe = colorframe
        self.intrinsics = rs_intrinsics

class CameraIntrinsics:
    def setRGBIntrinsics(self,RGBIntrinsics):
        self.cm = np.array(
            [[RGBIntrinsics.fx, 0, RGBIntrinsics.ppx], [0, RGBIntrinsics.fy, RGBIntrinsics.ppy], [0, 0, 1]])
        self.coeff = np.array(RGBIntrinsics.coeffs)


class Cameras:
    def __init__(self):
        self.ctx = rs.context()
        self.serials={}
        self.devices = self.ctx.query_devices()
        self.pipelines={}
        self.intrinsics={}
        self.align = rs.align(rs.stream.color)

        for item in self.devices:
            serial = item.get_info(rs.camera_info.serial_number)
            self.serials[len(self.serials)]=serial


    def captureRGB(self,index):
        pipeline = rs.pipeline(self.ctx)
        cfg = rs.config()
        cfg.enable_device(self.serials[index])
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        c=pipeline.start(cfg)
        self.pipelines[self.serials[index]]=pipeline

        rgb_profile = c.get_stream(rs.stream.color)
        rgb_intrinsic = rgb_profile.as_video_stream_profile().get_intrinsics()

        cameraintrinsics=CameraIntrinsics()
        cameraintrinsics.setRGBIntrinsics(rgb_intrinsic)
        #self.intrinsics[self.serials[index]]=cameraintrinsics
        self.intrinsics[self.serials[index]] = rgb_intrinsic
    
    def captureRGBwithResolution(self,index,width,height):
        pipeline = rs.pipeline(self.ctx)
        cfg = rs.config()
        cfg.enable_device(self.serials[index])
        cfg.enable_stream(rs.stream.color, width, height, rs.format.bgr8, 30)
        c=pipeline.start(cfg)
        self.pipelines[self.serials[index]]=pipeline

        rgb_profile = c.get_stream(rs.stream.color)
        rgb_intrinsic = rgb_profile.as_video_stream_profile().get_intrinsics()

        cameraintrinsics=CameraIntrinsics()
        cameraintrinsics.setRGBIntrinsics(rgb_intrinsic)
        #self.intrinsics[self.serials[index]]=cameraintrinsics
        self.intrinsics[self.serials[index]] = rgb_intrinsic


    def captureRGBandDepth(self,index):
        pipeline = rs.pipeline(self.ctx)
        cfg = rs.config()
        cfg.enable_device(self.serials[index])
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        c = pipeline.start(cfg)
        self.pipelines[self.serials[index]] = pipeline

        rgb_profile = c.get_stream(rs.stream.color)
        rgb_intrinsic = rgb_profile.as_video_stream_profile().get_intrinsics()

        cameraintrinsics = CameraIntrinsics()
        cameraintrinsics.setRGBIntrinsics(rgb_intrinsic)
        #self.intrinsics[self.serials[index]] = cameraintrinsics
        self.intrinsics[self.serials[index]] = rgb_intrinsic



    def getFrameset(self,index):
        return self.pipelines[self.serials[index]].wait_for_frames()

    def getRGBFrame(self,index):
        frameset = self.pipelines[self.serials[index]].wait_for_frames()
        color=frameset.get_color_frame()
        color=np.asanyarray(color.get_data())
        return color

    def getDepthFrame(self,index):
        pass

    def getRGBandDepthFrame(self,index):
        frameset = self.pipelines[self.serials[index]].wait_for_frames()
        aligned_frameset=self.align.process(frameset)
        color = aligned_frameset.get_color_frame()
        color = np.asanyarray(color.get_data())
        colorImg=ColorImg(colorframe=color,rs_intrinsics=self.intrinsics[self.serials[index]])

        depth = aligned_frameset.get_depth_frame()

        return (colorImg,depth)




if __name__ == '__main__':
    cams=Cameras()
    cams.captureRGB(0)
    while(True):
        img=cams.getRGBFrame(0)
        cv2.imshow('img',img)
        cv2.waitKey(1)
