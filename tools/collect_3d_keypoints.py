import json
import argparse
import torch
import trimesh
import os
from os import path as osp
from glob import glob
import numpy as np
from mmcv.ops import furthest_point_sample





def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', default='data/ycbv/models/models', type=str)
    parser.add_argument('--save-dir', default='data/ycbv/bbox.json', type=str)
    parser.add_argument('--type', choices=['bbox_oriented', 'bbox', 'FPS'], default='bbox')
    parser.add_argument('--keypoint-num', default=8, type=int, help='specific the keypoint num when using FPS')
    parser.add_argument('--category-specific', action='store_true')
    args = parser.parse_args()
    return args




if __name__ == '__main__':
    args = parse_args()
    models = glob(osp.join(args.model_dir, '*.ply'))
    models = sorted(models)
    keypoints_collect = []
    for model in models:
        model = trimesh.load(model)
        if args.type == 'bbox':
            keypoints = model.bounding_box.vertices.tolist()
        elif args.type == 'bbox_oriented':
            keypoints = model.bounding_box_oriented.vertices.tolist()
        else:
            vertices = torch.from_numpy(model.vertices.view(np.ndarray).astype(np.float32)).to('cuda').reshape(1, -1, 3)
            idx = furthest_point_sample(vertices, args.keypoint_num)[0]
            keypoints = vertices[0, idx.to(torch.int64), :].cpu().tolist()
        keypoints_collect.append(keypoints)
    if not osp.exists(args.save_dir):
        os.makedirs(args.save_dir, exist_ok=True)
    if args.category_specific:
        for i, model in enumerate(models):
            model_name = osp.basename(model).split('.')[0]
            save_path = osp.join(args.save_dir, model_name+'_'+args.type+'.json')
            if not osp.exists(osp.dirname(save_path)):
                os.makedirs(osp.dirname(save_path))
            with open(save_path, 'w') as f:
                json.dump(keypoints_collect[i], f, indent=4)
    save_path = osp.join(args.save_dir, args.type+f'_{args.keypoint_num}.json') \
                if args.type=='FPS' else osp.join(args.save_dir, args.type+f'.json')
    with open(save_path, 'w') as f:
        json.dump(keypoints_collect, f, indent=4)