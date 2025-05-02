import os
import cv2
import argparse
import os.path as osp
import csv
from io import BytesIO
from copy import deepcopy

import onnx
import onnxsim
import torch
import numpy as np
import supervision as sv
from PIL import Image
from torchvision.ops import nms
from mmengine.config import Config, ConfigDict, DictAction
from mmengine.runner import Runner
from mmengine.runner.amp import autocast
from mmengine.dataset import Compose
from mmdet.datasets import CocoDataset
from mmyolo.registry import RUNNERS

BOUNDING_BOX_ANNOTATOR = sv.BoundingBoxAnnotator()
LABEL_ANNOTATOR = sv.LabelAnnotator(text_color=sv.Color.BLACK)

def parse_args():
    parser = argparse.ArgumentParser(description='YOLO-World Batch Prediction')
    parser.add_argument('--config', default='configs/pretrain/yolo_world_xl_t2i_bn_2e-4_100e_4x8gpus_obj365v1_goldg_train_lvis_minival.py', help='test config file path')
    parser.add_argument('--checkpoint', default='weights/yolo_world_v2_xl_obj365v1_goldg_cc3mlite_pretrain.pth', help='checkpoint file')
    parser.add_argument('input_dir', help='directory with images to process')
    parser.add_argument('output_dir', help='directory to save processed images')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file.')
    args = parser.parse_args()
    return args

#delete outer boxes
def delete_obox(bboxes, iou_thread = 0.99):
    keep = []
    N = len(bboxes)
    for i in range(N):
        xi1, yi1, xi2, yi2 = bboxes[i]
        is_outer = False
        for j in range(N):
            if i == j:
                continue
            xj1, yj1, xj2, yj2 = bboxes[j]
            # 如果 i 包含 j，则 i 是外层框，应删除
            if xi1 <= xj1 and yi1 <= yj1 and xi2 >= xj2 and yi2 >= yj2:
                is_outer = True
                break
        if not is_outer:
            keep.append(i)
    return keep

def run_image(runner, image_path, text, max_num_boxes, score_thr, nms_thr, output_path):
    image = Image.open(image_path)
    texts = [[t.strip()] for t in text.split(',')] + [[' ']]
    data_info = dict(img_id=0, img_path=image_path, texts=texts)
    data_info = runner.pipeline(data_info)
    data_batch = dict(inputs=data_info['inputs'].unsqueeze(0),
                      data_samples=[data_info['data_samples']])

    with autocast(enabled=False), torch.no_grad():
        output = runner.model.test_step(data_batch)[0]
        pred_instances = output.pred_instances

    keep = nms(pred_instances.bboxes, pred_instances.scores, iou_threshold=nms_thr)
    pred_instances = pred_instances[keep]
    pred_instances = pred_instances[pred_instances.scores.float() > score_thr]

    if len(pred_instances.scores) > max_num_boxes:
        indices = pred_instances.scores.float().topk(max_num_boxes)[1]
        pred_instances = pred_instances[indices]
        #delete outer boxes
        
    indices = delete_obox(pred_instances.bboxes)
    pred_instances = pred_instances[indices]
    
    pred_instances = pred_instances.cpu().numpy()
    detections = sv.Detections(
        xyxy=pred_instances['bboxes'],
        class_id=pred_instances['labels'],
        confidence=pred_instances['scores']
    )
    
    # 将类别标签和分数组合成 (label, score) 形式的字符串
    labels_with_scores = [
        f"{texts[class_id][0]} ({confidence:.2f}): {bounding_box}"
        for class_id, confidence, bounding_box in zip(detections.class_id, detections.confidence, detections.xyxy)
    ]
    
    # 保存带注释的图像
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    image = BOUNDING_BOX_ANNOTATOR.annotate(image, detections)
    image = LABEL_ANNOTATOR.annotate(image, detections, labels=labels_with_scores)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    output_image = Image.fromarray(image)
    output_image.save(output_path)
    
    # 返回检测到的类别和分数的列表
    return labels_with_scores

if __name__ == '__main__':
    args = parse_args()

    # 加载配置
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    os.makedirs(args.output_dir, exist_ok=True)

    cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    cfg.load_from = args.checkpoint

    if 'runner_type' not in cfg:
        runner = Runner.from_cfg(cfg)
    else:
        runner = RUNNERS.build(cfg)

    runner.call_hook('before_run')
    runner.load_or_resume()
    pipeline = cfg.test_dataloader.dataset.pipeline
    runner.pipeline = Compose(pipeline)
    runner.model.eval()

    with open('classes.txt', 'r') as file:
        text = file.read()
    
    # 准备CSV文件，用于保存检测到的类别和分数信息
    csv_path = osp.join(args.output_dir, 'part6.csv')
    with open(csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Image', 'Classes and Scores'])  # 写入CSV文件的头部

        # 处理输入目录中的每张图片
        for filename in os.listdir(args.input_dir):
            if filename.endswith(('.jpg', '.jpeg', '.png')):
                input_path = os.path.join(args.input_dir, filename)
                output_path = os.path.join(args.output_dir, filename)
                
                # 运行图像检测并获取类别和分数信息
                labels_with_scores = run_image(
                    runner, input_path, text=text, max_num_boxes=100, score_thr=0.05, nms_thr=0.6, output_path=output_path
                )
                
                # 将类别和分数信息写入CSV文件，每张图片的信息放在一行，用逗号分隔
                csv_writer.writerow([filename, ", ".join(labels_with_scores)])
                print(f"Processed {filename} and saved to {output_path}")

    print(f"Detection classes and scores saved to {csv_path}")

