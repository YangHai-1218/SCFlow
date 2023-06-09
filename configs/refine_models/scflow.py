_base_ = '../refine_datasets/ycbv_real.py'

dataset_root = 'data/ycbv'

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

model = dict(
    type='SCFlowRefiner',
    cxt_channels=128,
    h_channels=128,
    seperate_encoder=False,
    max_flow=400.,
    filter_invalid_flow=True,
    encoder=dict(
        type='RAFTEncoder',
        in_channels=3,
        out_channels=256,
        net_type='Basic',
        norm_cfg=dict(type='IN'),
        init_cfg=[
            dict(
                type='Kaiming',
                layer=['Conv2d'],
                mode='fan_out',
                nonlinearity='relu'),
            dict(type='Constant', layer=['InstanceNorm2d'], val=1, bias=0)
        ]),
    cxt_encoder=dict(
        type='RAFTEncoder',
        in_channels=3,
        out_channels=256,
        net_type='Basic',
        norm_cfg=dict(type='BN'),
        init_cfg=[
            dict(
                type='Kaiming',
                layer=['Conv2d'],
                mode='fan_out',
                nonlinearity='relu'),
            dict(type='Constant', layer=['SyncBatchNorm2d'], val=1, bias=0)
        ]),
    decoder=dict(
        type='SCFlowDecoder',
        net_type='Basic',
        num_levels=4,
        radius=4,
        iters=8,
        detach_flow=True,
        detach_mask=True,
        detach_pose=True,
        detach_depth_for_xy=True,
        mask_flow=False,
        mask_corr=False,
        pose_head_cfg=dict(
            type='MultiClassPoseHead',
            num_class=21,
            in_channels=224,
            net_type='Basic',
            rotation_mode='ortho6d',
            norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
            act_cfg=dict(type='ReLU'),
        ),
        corr_lookup_cfg=dict(align_corners=True),
        gru_type='SeqConv',
        act_cfg=dict(type='ReLU')),
    flow_loss_cfg=dict(
        type='SequenceLoss',
        gamma=0.8,
        loss_func_cfg=dict(
            type='RAFTLoss',
            loss_weight=.1,
            max_flow=400.,
        )
    ),
    pose_loss_cfg=dict(
        type='SequenceLoss',
        gamma=0.8,
        loss_func_cfg=dict(
            type='DisentanglePointMatchingLoss',
            symmetry_types=symmetry_types,
            mesh_diameter=mesh_diameter,
            mesh_path=dataset_root+'/models_eval',
            loss_type='l1',
            disentangle_z=True,
            loss_weight=10.0,
        )
    ),
    mask_loss_cfg=dict(
        type='SequenceLoss',
        gamma=0.8,
        loss_func_cfg=dict(
            type='L1Loss',
            loss_weight=10.,
        )
    ),
    freeze_bn=False,
    freeze_encoder=False,
    train_cfg=dict(),
    test_cfg=dict(iters=8),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='work_dirs/raft_8x2_100k_flyingthings3d_400x720_convertered.pth'
    )
)



optimizer = dict(
    type='AdamW',
    lr=0.0004,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.0001,
    amsgrad=False,
    )
optimizer_config = dict(grad_clip=dict(max_norm=10.))
lr_config = dict(
    policy='OneCycle',
    max_lr=0.0004,
    total_steps=100100,
    pct_start=0.05,
    anneal_strategy='linear')
evaluation=dict(interval=5000, 
                metric={
                    'auc':[],
                    'add':[0.05, 0.10, 0.20, 0.50]},
                save_best='average/add_10',
                rule='greater'
            )
runner = dict(type='IterBasedRunner', max_iters=100000)
num_gpus = 1
checkpoint_config = dict(interval=10000, by_epoch=False)
log_config=dict(interval=50, 
                hooks=[
                    dict(type='TextLoggerHook'),
                    dict(type='TensorboardImgLoggerHook', interval=100, image_format='HWC')])
work_dir = 'work_dirs/scflow_ycbv_real'