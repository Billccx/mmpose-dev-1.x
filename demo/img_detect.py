import mimetypes
import os
import tempfile
from argparse import ArgumentParser

import mmcv
import mmengine
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples
from mmpose.utils import register_all_modules as register_mmpose_modules

from mmdet.apis import inference_detector, init_detector
from mmdet.utils import register_all_modules as register_mmdet_modules

import cv2

'''
python demo/topdown_demo_with_mmdet_my.py 
demo/mmdetection_cfg/cascade_rcnn_x101_64x4d_fpn_1class.py 
demo/checkpoints/cascade_rcnn_x10/td-hm_mobilenetv2_8xb64-210e_onehand10k-256x256.py 
demo/checkpoints/mobilenetv2_onehand10k_256x256-f3a3d90e_20210330.pth 
--input /home/ccx/code/python/mmpose-1.x/demo/video/u1.mov 
--output-root /home/ccx/code/python/mmpose-1.x/demo/outputs 
--draw-heatmap
'''

handSkeleton=[[0,1],[1,2],[2,3],[3,4],
              [0,5],[5,6],[6,7],[7,8],[5,9],
              [9,10],[10,11],[11,12],[9,13],
              [13,14],[14,15],[15,16],[13,17],
              [0,17],[17,18],[18,19],[19,20]]

bodySkeleton=[]

def drawSkeleton(img,points,skeleton):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 1, (255, 0, 0), 2)

    for line in skeleton:
        x0 = int(points[line[0]][0])
        y0 = int(points[line[0]][1])

        x1 = int(points[line[1]][0])
        y1 = int(points[line[1]][1])

        cv2.line(img, (x0, y0), (x1, y1), (0,255,0),1)


def main():
    handdet_config='demo/mmdetection_cfg/ssdlite_mobilenetv2_scratch_600e_onehand.py'
    handdet_checkpoint='demo/checkpoints/ssdlite_mobilenetv2_scratch_600e_onehand-4f9f8686_20220523.pth'

    handpose_config='demo/pose_estimate_cfg/td-hm_mobilenetv2_8xb64-210e_onehand10k-256x256.py'
    handpose_checkpoint='demo/checkpoints/mobilenetv2_onehand10k_256x256-f3a3d90e_20210330.pth'

    register_mmdet_modules()
    hand_detector = init_detector(handdet_config, handdet_checkpoint, device='cuda:0')

    register_mmpose_modules()
    handpose_estimator = init_pose_estimator(handpose_config,handpose_checkpoint,device='cuda:0')

    thr=0.3

    img=cv2.imread("/home/cuichenxi/code/Python/mmpose-dev-1.x/demo/video/2.png")
    #img="/home/cuichenxi/code/Python/mmpose-dev-1.x/demo/video/2.png"

    #hand detect
    register_mmdet_modules()
    detect_result = inference_detector(hand_detector, img)
    pred_instance = detect_result.pred_instances.cpu().numpy()
    bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
    bboxes = bboxes[np.logical_and(pred_instance.labels == 0,pred_instance.scores > thr)]
    bboxes = bboxes[nms(bboxes, thr)][:, :4]


    #hand estimate
    register_mmpose_modules()
    pose_results = inference_topdown(handpose_estimator, img, bboxes)
    data_samples = merge_data_samples(pose_results)

    hands=[]
    for i in range(len(pose_results)):
        hand={}
        scores=pose_results[i].pred_instances.keypoint_scores
        scores=scores.reshape([scores.shape[0],scores.shape[1],1])
        keypoints=pose_results[i].pred_instances.keypoints
        kps_score=np.concatenate((keypoints,scores),axis=2).squeeze(0)
        bbox=pose_results[i].pred_instances.bboxes[0]
        hand['kps']=kps_score
        hand['bbox']=bbox
        hands.append(hand)
        print(hand)

        cv2.circle(img, (int(kps_score[0][0]), int(kps_score[0][1])), 2, (255, 0, 0), 2)
        drawSkeleton(img,kps_score,handSkeleton)

    for i in range(len(bboxes)):
        cv2.rectangle(img,(int(bboxes[i][0]),int(bboxes[i][1])),(int(bboxes[i][2]),int(bboxes[i][3])),(255,0,0),2)


    cv2.imshow('img',img)
    cv2.waitKey(0)




if __name__=='__main__':
    main()