import mimetypes
import os
import tempfile
import mimetypes
import os
import tempfile
from argparse import ArgumentParser

import json_tricks as json
import mmcv
import mmengine
import numpy as np

from mmpose.apis import inference_topdown
from mmpose.apis import init_model as init_pose_estimator
from mmpose.evaluation.functional import nms
from mmpose.registry import VISUALIZERS
from mmpose.structures import merge_data_samples, split_instances
from mmpose.utils import register_all_modules as register_mmpose_modules

from mmyolo.utils import register_all_modules as register_mmyolo_modules

try:
    from mmdet.apis import inference_detector, init_detector
    from mmdet.utils import register_all_modules as register_mmdet_modules
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

import cv2
from cameras import Cameras

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

bodySkeleton=[[15,13],[13,11],[16,14],[14,12],
              [11,12],[5,11],[6,12], [5,6],[5,7],
              [6,8],[7,9],[8,10],[1,2],[0,1],
              [0,2],[1,3],[2,4],[3,5],[4,6]]

def drawSkeleton(img,points,skeleton,rgb):
    for point in points:
        cv2.circle(img, (int(point[0]), int(point[1])), 1, (255, 0, 0), 2)

    for line in skeleton:
        x0 = int(points[line[0]][0])
        y0 = int(points[line[0]][1])

        x1 = int(points[line[1]][0])
        y1 = int(points[line[1]][1])

        cv2.line(img, (x0, y0), (x1, y1), rgb,1)


def drawBBoxes(img,bboxes,rgb):
    for i in range(len(bboxes)):
        cv2.rectangle(img, (int(bboxes[i][0]), int(bboxes[i][1])),
                      (int(bboxes[i][2]), int(bboxes[i][3])),
                      rgb, 2)



def processResult(pose_results):
    results=[]
    for i in range(len(pose_results)):
        result = {}
        scores = pose_results[i].pred_instances.keypoint_scores
        scores = scores.reshape([scores.shape[0], scores.shape[1], 1])
        keypoints = pose_results[i].pred_instances.keypoints
        kps_score = np.concatenate((keypoints, scores), axis=2).squeeze(0)
        bbox = pose_results[i].pred_instances.bboxes[0]
        result['kps'] = kps_score
        result['bbox'] = bbox
        results.append(result)
    return results




def main():
    humandet_config='demo/yolo_cfg/yolov7_tiny_syncbn_fast_8x16b-300e_coco.py'
    humandet_checkpoint='demo/checkpoints/yolov7_tiny_syncbn_fast_8x16b-300e_coco_20221126_102719-0ee5bbdf.pth'

    humanpose_config='configs/body_2d_keypoint/topdown_heatmap/coco/td-hm_vipnas-mbv3_8xb64-210e_coco-256x192.py'
    humanpose_checkpoint='demo/checkpoints/vipnas_mbv3_coco_256x192-7018731a_20211122.pth'

    handdet_config='demo/mmdetection_cfg/ssdlite_mobilenetv2_scratch_600e_onehand.py'
    handdet_checkpoint='demo/checkpoints/ssdlite_mobilenetv2_scratch_600e_onehand-4f9f8686_20220523.pth'

    handpose_config='demo/pose_estimate_cfg/td-hm_mobilenetv2_8xb64-210e_onehand10k-256x256.py'
    handpose_checkpoint='demo/checkpoints/mobilenetv2_onehand10k_256x256-f3a3d90e_20210330.pth'


    register_mmdet_modules()
    hand_detector = init_detector(handdet_config, handdet_checkpoint, device='cuda:0')

    register_mmpose_modules()
    handpose_estimator = init_pose_estimator(
        handpose_config,
        handpose_checkpoint,
        device='cuda:0',
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=False))
        )
    )


    #register_mmdet_modules()
    register_mmyolo_modules()
    human_detector = init_detector(humandet_config, humandet_checkpoint, device='cuda:0')

    register_mmpose_modules()
    humanpose_estimator = init_pose_estimator(
        humanpose_config,
        humanpose_checkpoint,
        device='cuda:0',
        cfg_options=dict(
            model=dict(test_cfg=dict(output_heatmaps=False))
        )
    )

    # handpose_estimator.cfg.visualizer.radius = 3
    # handpose_estimator.cfg.visualizer.line_width = 1
    # visualizer = VISUALIZERS.build(handpose_estimator.cfg.visualizer)
    # # the dataset_meta is loaded from the checkpoint and
    # # then pass to the model in init_pose_estimator
    # visualizer.set_dataset_meta(handpose_estimator.dataset_meta)

    #video = cv2.VideoCapture(0)
    #video = cv2.VideoCapture('/home/cuichenxi/code/Python/mmpose-dev-1.x/demo/video/test.mp4')
    #writer= cv2.VideoWriter('result.mp4', cv2.VideoWriter_fourcc(*'DIVX'), 20, (1280,720))

    cams=Cameras()
    cams.captureRGBwithResolution(0,1280,720)

    thr=0.3

    while True:
        img=cams.getRGBFrame(0)

        # hand detect
        register_mmdet_modules()
        detect_result = inference_detector(hand_detector, img)
        pred_instance = detect_result.pred_instances.cpu().numpy()
        hand_bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        hand_bboxes = hand_bboxes[np.logical_and(pred_instance.labels == 0, pred_instance.scores > thr)]
        hand_bboxes = hand_bboxes[nms(hand_bboxes, thr)][:, :4]

        # hand estimate
        register_mmpose_modules()
        hand_pose_results = inference_topdown(handpose_estimator, img, hand_bboxes)
        data_samples = merge_data_samples(hand_pose_results)



        # human detect
        #register_mmdet_modules()
        register_mmyolo_modules()
        humandetect_result = inference_detector(human_detector, img)
        human_pred_instance = humandetect_result.pred_instances.cpu().numpy()
        human_bboxes = np.concatenate((human_pred_instance.bboxes, human_pred_instance.scores[:, None]), axis=1)
        human_bboxes = human_bboxes[np.logical_and(human_pred_instance.labels == 0, human_pred_instance.scores > thr)]
        human_bboxes = human_bboxes[nms(human_bboxes, thr)][:, :4]

        # human estimate
        register_mmpose_modules()
        human_pose_results = inference_topdown(humanpose_estimator, img, human_bboxes)


        hands=processResult(hand_pose_results)
        humans=processResult(human_pose_results)

        drawBBoxes(img, human_bboxes,(0,0,255))
        drawBBoxes(img, hand_bboxes,(0,255,255))

        for hand in hands:
            drawSkeleton(img, hand['kps'], handSkeleton,(255,0,255))
        for human in humans:
            drawSkeleton(img, human['kps'], bodySkeleton,(255,0,255))

        cv2.imshow('img', img)
        cv2.waitKey(1)

        '''
        for i in range(len(human_bboxes)):
            cv2.rectangle(img, (int(human_bboxes[i][0]), int(human_bboxes[i][1])), (int(human_bboxes[i][2]), int(human_bboxes[i][3])),
                          (255, 0, 0), 2)


        hands = []
        for i in range(len(hand_pose_results)):
            hand = {}
            scores = hand_pose_results[i].pred_instances.keypoint_scores
            scores = scores.reshape([scores.shape[0], scores.shape[1], 1])
            keypoints = hand_pose_results[i].pred_instances.keypoints
            kps_score = np.concatenate((keypoints, scores), axis=2).squeeze(0)
            bbox = hand_pose_results[i].pred_instances.bboxes[0]
            hand['kps'] = kps_score
            hand['bbox'] = bbox
            hands.append(hand)

            cv2.circle(img, (int(kps_score[0][0]), int(kps_score[0][1])), 2, (255, 0, 0), 2)
            drawSkeleton(img, kps_score, handSkeleton)

        for i in range(len(hand_bboxes)):
            cv2.rectangle(img, (int(hand_bboxes[i][0]), int(hand_bboxes[i][1])), (int(hand_bboxes[i][2]), int(hand_bboxes[i][3])),
                          (255, 0, 0), 2)
        '''

        '''
        #hand detect
        register_mmdet_modules()
        detect_result = inference_detector(hand_detector, img)
        pred_instance = detect_result.pred_instances.cpu().numpy()
        #print(pred_instance)
        bboxes = np.concatenate((pred_instance.bboxes, pred_instance.scores[:, None]), axis=1)
        bboxes = bboxes[np.logical_and(pred_instance.labels == 0,pred_instance.scores > thr)]
        bboxes = bboxes[nms(bboxes, thr)][:, :4]
        #print(bboxes,cnt)
        cnt+=1
        print(len(bboxes))
        for i in range(len(bboxes)):
            cv2.rectangle(img,(int(bboxes[i][0]),int(bboxes[i][1])),(int(bboxes[i][2]),int(bboxes[i][3])),(255,0,0),2)


        #hand estimate
        register_mmpose_modules()
        pose_results = inference_topdown(handpose_estimator, img, bboxes)
        data_samples = merge_data_samples(pose_results)
        #print(data_samples)

        hands=[]
        for i in range(len(data_samples.bbox_scores)):
            print(data_samples.keypoints[i].shape)
            print(data_samples.keypoint_scores[i].shape)
            #keypoints = np.concatenate((data_samples.keypoints[i], data_samples.keypoint_scores[:, None]), axis=1)



        cv2.imshow('img',img)
        cv2.waitKey(1)
        '''


if __name__=='__main__':
    main()