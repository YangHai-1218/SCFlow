import mmcv
import cv2
import random
import argparse
import numpy as np
from mmcv.utils import Config
from datasets import build_dataset
from pathlib import Path
from tools.visualize import imshow_projected_points, imshow_pose_axis, imshow_pose_contour
from multiprocessing import Pool

from tools.visualize import imshow_2d_keypoints

colors = ['red', 'green', 'blue', 'cyan', 'yellow', 'magenta', 'white', 'black']*10
def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset on patch level')
    parser.add_argument('--config', default='configs/refine_datasets/ycbv_pbr_adaptk.py', help='train config file path')
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['Normalize', 'ToTensor', 'Collect'],
        help='skip some useless pipelines to better visualize')
    parser.add_argument(
        '--output-dir',
        default='temp/',
        type=str,
        help='If there is no display interface, you can save it'
    )
    parser.add_argument(
        '--dataset-type',
        default='val',
        type=str,
    )
    parser.add_argument(
        '--show',
        default='contour',
        nargs='?',
        choices=['project', 'axis', 'contour', 'keypoints_2d'],
    )
    parser.add_argument(
        '--not-show', default=False, action='store_true',
    )
    parser.add_argument(
        '--show-interval', type=float, default=2, help='the interval of show (s)'
    )
    parser.add_argument(
        '--random', action='store_true', help='randomly choose data',
    )
    parser.add_argument(
        '--multi-process', default=-1, type=int, help='multi-processsing to speed up'
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


def visualize_func(dataset, index, args, meshes):
    item = dataset[index] 
    patch_num = len(item['img'])
    for i in range(patch_num):
        sequence = str(Path(item['img_path']).parents[1].name)
        filename = Path(args.output_dir).joinpath(sequence + '_' + str(Path(item['img_path']).stem) + f"_{i:06d}.png") \
            if args.output_dir is not None else None
        filename = str(filename)
        if args.show == 'project':
            show_image = []
            if 'gt_rotations' in item:
                gt_image = imshow_projected_points(
                    item['img'][i], meshes, item['gt_rotations'][i][None], item['gt_translations'][i][None], item['labels'][i][None], item['k'][i][None],
                    class_names=dataset.class_names, out_file=None, show = not args.not_show,
                )
                show_image.append(gt_image)
            if 'ref_rotations' in item:
                ref_image = imshow_projected_points(
                    item['img'][i], meshes, item['ref_rotations'][i][None], item['ref_translations'][i][None], item['labels'][i][None], item['k'][i][None],
                    class_names=dataset.class_names, out_file=None, show = not args.not_show,
                )
                show_image.append(ref_image)
            image = cv2.hconcat(show_image)
            cv2.imwrite(filename, image)
        elif args.show == 'axis':
            show_image = []
            if 'gt_rotations' in item:
                gt_image = imshow_pose_axis(
                    item['img'][i], dataset.mesh_diameter/2, item['gt_rotations'][i][None], item['gt_translations'][i][None], item['labels'][i][None], item['k'][i][None],
                    class_names=dataset.class_names, out_file=None, show = not args.not_show,
                )
                show_image.append(gt_image)
            if 'ref_rotations' in item:
                ref_image = imshow_pose_axis(
                    item['img'][i], dataset.mesh_diameter/2, item['ref_rotations'][i][None], item['ref_translations'][i][None], item['labels'][i][None], item['k'][i][None],
                    class_names=dataset.class_names, out_file=None, show = not args.not_show,
                )
                show_image.append(ref_image)
            image = cv2.hconcat(show_image)
            cv2.imwrite(filename, image)
            # image = imshow_pose_axis(
            #     image, dataset.mesh_diameter/2, item['gt_rotations'][i][None], item['gt_translations'][i][None], item['labels'][i][None], item['k'][i][None],
            #     class_names=dataset.class_names, out_file=filename, show = not args.not_show,
            # )
        elif args.show == 'contour':
            show_image = []
            if 'gt_rotations' in item:
                gt_image = imshow_pose_contour(
                    item['img'][i], dataset.meshes, item['gt_rotations'][i][None], item['gt_translations'][i][None], item['labels'][i][None], item['k'][i][None],
                    class_names=dataset.class_names, out_file=filename, show = not args.not_show,
                )
                show_image.append(gt_image)
            if 'ref_rotations' in item:
                ref_image = imshow_pose_contour(
                    item['img'][i], dataset.meshes, item['ref_rotations'][i][None], item['ref_translations'][i][None], item['labels'][i][None], item['k'][i][None],
                    class_names=dataset.class_names, out_file=None, show = not args.not_show
                )
                show_image.append(ref_image)
            image = cv2.hconcat(show_image)
            cv2.imwrite(filename, image)
        elif args.show == 'keypoints_2d':
            image = imshow_2d_keypoints(
                item['img'][i], item['gt_keypoints_2d'][i][None], item['labels'][i][None],
                order=[(0, 1), (0, 2), (1, 3), (2, 3),(0, 4), (1, 5), (3, 7), (2, 6), (4, 5), (4, 6), (5, 7), (6, 7)], 
                class_names=dataset.class_names, show = not args.not_show
            )


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type, args.dataset_type)
    
    dataset = build_dataset(cfg.data.get(args.dataset_type))

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
    
    if args.multi_process > 0:
        pool = Pool(args.multi_process)
        for index in indices:
            pool.apply_async(visualize_func, args=(dataset, index, args, meshes,))
        pool.close()
        pool.join()
    else:
        for index in indices:
            visualize_func(dataset, index, args, meshes)
            progress_bar.update()




if __name__ == '__main__':
    main()
