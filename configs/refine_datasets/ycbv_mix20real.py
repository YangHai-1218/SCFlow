dataset_root = 'data/ycbv'

CLASS_NAMES= ('master_chef_can', 'cracker_box',
            'sugar_box', 'tomato_soup_can',
            'mustard_bottle', 'tuna_fish_can',
            'pudding_box', 'gelatin_box',
            'potted_meat_can', 'banana',
            'pitcher_base', 'bleach_cleanser',
            'bowl', 'mug', 'power_drill', 
            'wood_block', 'scissors', 'large_marker',
            'large_clamp', 'extra_large_clamp', 'foam_brick')
normalize_mean = [0., 0., 0., ]
normalize_std = [255., 255., 255.]
image_scale = 256
symmetry_types = { # 1-base
    'cls_13': {'z':0},
    'cls_16': {'x':180, 'y':180, 'z':90},
    'cls_19': {'y':180},
    'cls_20': {'x':180},
    'cls_21': {'x':180, 'y':90, 'z':180}
}
mesh_diameter = [172.16, 269.58, 198.38, 120.66, 199.79, 90.17, 142.58, 114.39, 129.73,
                198.40, 263.60, 260.76, 162.27, 126.86, 230.44, 237.30, 204.11, 121.46,
                183.08, 231.39, 102.92]
file_client_args = dict(backend='disk',)

train_pipeline = [
    dict(type='LoadImages', color_type='unchanged', file_client_args=file_client_args),
    dict(type='LoadMasks', file_client_args=file_client_args),
    dict(type='PoseJitter',
        jitter_angle_dis=(0, 15),
        jitter_x_dis=(0, 15),
        jitter_y_dis=(0, 15),
        jitter_z_dis=(0, 50),
        angle_limit=45, 
        translation_limit=200,
        add_limit=1.,
        mesh_dir=dataset_root + '/models_eval',
        mesh_diameter=mesh_diameter,
        jitter_pose_field=['gt_rotations', 'gt_translations'],
        jittered_pose_field=['ref_rotations', 'ref_translations']),
    dict(type='ComputeBbox', mesh_dir=dataset_root + '/models_eval', clip_border=False),
    dict(type='Crop',
        size_range=(1.0, 1.25), 
        crop_bbox_field='ref_bboxes',
        clip_border=False,
        pad_val=128,
    ),
    dict(type='RandomBackground', background_dir='data/coco', p=0.3, file_client_args=file_client_args),
    dict(type='RandomHSV', h_ratio=0.2, s_ratio=0.5, v_ratio=0.5),
    dict(type='RandomNoise', noise_ratio=0.1),
    dict(type='RandomSmooth', max_kernel_size=5.),
    dict(type='Resize', img_scale=image_scale, keep_ratio=True),
    dict(type='Pad', size=(image_scale, image_scale), center=True, pad_val=dict(img=(128, 128, 128), mask=0)),
    dict(type='RemapPose', keep_intrinsic=False),
    dict(type='Normalize', mean=normalize_mean, std=normalize_std, to_rgb=True),
    dict(type='ToTensor', stack_keys=[], ),
    dict(type='Collect', 
        annot_keys=[
            'ref_rotations', 'ref_translations', 
            'gt_rotations', 'gt_translations', 'gt_masks',
            'init_add_error', 'init_rot_error', 'init_trans_error',
            'k', 'labels'],
        meta_keys=(
            'img_path', 'ori_shape', 'ori_k',
            'img_shape', 'img_norm_cfg', 
            'scale_factor', 'transform_matrix',
            'ori_gt_rotations', 'ori_gt_translations'),
    ),
]

test_pipeline = [
    dict(type='LoadImages', color_type='unchanged', file_client_args=file_client_args),
    dict(type='ComputeBbox', mesh_dir=dataset_root + '/models_eval', clip_border=False),
    dict(type='Crop', 
        size_range=(1.1, 1.1),
        crop_bbox_field='ref_bboxes', 
        clip_border=False,
        pad_val=128),
    dict(type='Resize', img_scale=image_scale, keep_ratio=True),
    dict(type='Pad', size=(image_scale, image_scale), center=True, pad_val=dict(img=(128, 128, 128), mask=0)),
    dict(type='RemapPose', keep_intrinsic=False),
    dict(type='Normalize', mean=normalize_mean, std=normalize_std, to_rgb=True),
    dict(type='ToTensor', stack_keys=[], ),
    dict(type='Collect', 
        annot_keys=[
            'ref_rotations', 'ref_translations',
            'gt_rotations', 'gt_translations',
            'labels','k','ori_k','transform_matrix',
        ],
        meta_keys=(
            'img_path', 'ori_shape', 'img_shape', 'img_norm_cfg', 
            'scale_factor', 'keypoints_3d', 'geometry_transform_mode'),
    ),
]


data = dict(
    samples_per_gpu=24,
    workers_per_gpu=16,
    test_samples_per_gpu=1,
    train=dict(
        type='ConcatDataset',
        ratios=[100., 1.],
        dataset_configs=[
            dict(type='SuperviseTrainDataset',
                data_root=dataset_root + '/train_real',
                gt_annots_root=dataset_root + '/train_real',
                image_list=dataset_root + '/image_lists/train_real_20.txt',
                keypoints_json=dataset_root + '/keypoints/bbox.json',
                pipeline=train_pipeline,
                class_names=CLASS_NAMES,
                keypoints_num=8,
                sample_num=1,
                mesh_symmetry=symmetry_types,
                meshes_eval=dataset_root+'/models_eval',
                mesh_diameter=mesh_diameter,
            ),
            dict(type='SuperviseTrainDataset',
                data_root=dataset_root + '/train_pbr',
                gt_annots_root=dataset_root + '/train_pbr',
                image_list=dataset_root + '/image_lists/train_pbr.txt',
                keypoints_json=dataset_root + '/keypoints/bbox.json',
                pipeline=train_pipeline,
                class_names=CLASS_NAMES,
                sample_num=1,
                keypoints_num=8,
                min_visib_fract=0.2,
                mesh_symmetry=symmetry_types,
                meshes_eval=dataset_root+'/models_eval',
                mesh_diameter=mesh_diameter,)
        ]
    ),
    val=dict(
        type='RefineDataset',
        data_root=dataset_root + '/test',
        ref_annots_root='data/initial_poses/ycbv_posecnn',
        image_list=dataset_root + '/image_lists/test.txt',
        keypoints_json=dataset_root + '/keypoints/bbox.json',
        pipeline=test_pipeline,
        class_names=CLASS_NAMES,
        keypoints_num=8,
        mesh_symmetry=symmetry_types,
        meshes_eval=dataset_root+'/models_eval',
        mesh_diameter=mesh_diameter
    ),
    test=dict(
        type='RefineDataset',
        data_root=dataset_root + '/test',
        ref_annots_root='data/initial_poses/ycbv_posecnn',
        image_list=dataset_root + '/image_lists/test.txt',
        keypoints_json=dataset_root + '/keypoints/bbox.json',
        pipeline=test_pipeline,
        class_names=CLASS_NAMES,
        keypoints_num=8,
        mesh_symmetry=symmetry_types,
        meshes_eval=dataset_root+'/models_eval',
        mesh_diameter=mesh_diameter,
    ),
)

# renderer setting
model = dict(
    renderer=dict(
        mesh_dir=dataset_root + '/models_1024',
        image_size=(image_scale, image_scale),
        shader_type='Phong',
        soft_blending=False,
        render_mask=False,
        render_image=True,
        seperate_lights=True,
        faces_per_pixel=1,
        blur_radius=0.,
        sigma=1e-12,
        gamma=1e-12,
        background_color=(.5, .5, .5),
    ),
)