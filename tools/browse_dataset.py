from typing import Optional
import mmcv
import cv2
import random
import argparse
import numpy as np
from mmcv.utils import Config
from datasets import build_dataset
from pathlib import Path
from tools.visualize import imshow_projected_points, Pytorch3dVisTool

colors = ['green', 'blue', 'cyan', 'yellow', 'magenta', 'coral', 
        'purple', 'cornflowerblue', 'red', 'orangered', 'hotpink', 
        'darkgreen', 'blueviolet', 'orange']*10
def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset on the full image')
    parser.add_argument('--config', default='configs/bop_refine_datasets/lmo_pbr.py', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['ComputeBbox', 'Crop', 'Resize', 'Pad', 'RemapPose', 'Normalize', 'ToTensor', 'Collect'],
        help='skip some useless pipelines to better visualize')
    parser.add_argument(
        '--output-dir',
        default='debug/',
        type=str,
        help='If there is no display interface, you can save it'
    )
    parser.add_argument(
        '--dataset-type', default='test', type=str,
    )
    parser.add_argument(
        '--show',
        default='mask',
        nargs='?',
        choices=['project', 'axis', 'contour', 'keypoints_2d', 'mask'],
    )
    parser.add_argument(
        '--not-show', default=False, action='store_true',
    )
    parser.add_argument(
        '--random', action='store_true', help='randomly choose data',
    )
    args = parser.parse_args()
    return args

def retrieve_data_cfg(config_path, skip_type, dataset_type):

    def skip_pipeline_steps(config):
        config['pipeline'] = [
            x for x in config.pipeline if x['type'] not in skip_type
        ]

    cfg = Config.fromfile(config_path)
    # import modules from string list.
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    data_cfg = cfg.data.get(dataset_type)
    
    if data_cfg.type == 'ConcatDataset':
        for data_cfg in data_cfg.dataset_configs:
            skip_pipeline_steps(data_cfg)
    else:
        skip_pipeline_steps(data_cfg)

    return cfg


def visualize_func(dataset, index, args, meshes, vis_tool:Optional[Pytorch3dVisTool]=None):
    item = dataset[index]
    sequence = str(Path(item['img_path']).parents[1].name)
    filename = Path(args.output_dir).joinpath(sequence + '_' + str(Path(item['img_path']).stem)+'.png') \
        if args.output_dir is not None else None
    filename = str(filename)
    if args.show == 'project':
        show_image = []
        if 'gt_rotations' in item:
            gt_image = imshow_projected_points(
                item['img'], meshes, item['gt_rotations'], item['gt_translations'], item['labels'], item['k'], 
                class_names=dataset.class_names, out_file=None, show = not args.not_show,)
            show_image.append(gt_image)
        if 'ref_rotations' in item:
            ref_image = imshow_projected_points(
                item['img'], meshes, item['ref_rotations'], item['ref_translations'], item['labels'], item['k'],
                class_names=dataset.class_names, out_file=None, show = not args.not_show,)
            show_image.append(ref_image)
        show_image = cv2.hconcat(show_image)
        cv2.imwrite(filename, show_image)
    else:
        show_image = []
        if 'gt_rotations' in item:
            gt_image = vis_tool(item['img'], item['gt_rotations'], item['gt_translations'], item['labels'], item['k'], out_file=None)
            show_image.append(gt_image)
        if 'ref_rotations' in item:
            ref_image = vis_tool(item['img'], item['ref_rotations'], item['ref_translations'], item['labels'], item['k'], out_file=None)
            show_image.append(ref_image)
        show_image = cv2.hconcat(show_image)
        cv2.imwrite(filename, show_image)
    



def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.dataset_type)
    
    dataset_cfg = cfg.data.get(args.dataset_type)
    dataset = build_dataset(dataset_cfg)

    progress_bar = mmcv.ProgressBar(len(dataset))
    if args.output_dir is not None:
        mmcv.mkdir_or_exist(args.output_dir)
    
    if args.show == 'project':
        meshes = [mesh.vertices.view(np.ndarray) for mesh in dataset.meshes]
    else:
        meshes = None
    
        
    indices = list(range(len(dataset)))
    if args.random:
        random.shuffle(indices)
    if args.show in ['contour', 'mask']:
        vis_tool = Pytorch3dVisTool(cfg.model.renderer.mesh_dir, (480, 640), colors=colors, vis_mode=args.show)
    else:
        vis_tool = None
    for index in indices:
        visualize_func(dataset, index, args, meshes, vis_tool)
        progress_bar.update()




if __name__ == '__main__':
    main()
