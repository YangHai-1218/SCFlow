_base_ = '../refine_datasets/ycbv_real.py'


model = dict(
    type='RAFTRefinerFlowMask',
    cxt_channels=128,
    h_channels=128,
    seperate_encoder=False,
    max_flow=400.,
    filter_invalid_flow_by_mask=True,
    filter_invalid_flow_by_depth=False,
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
        type='RAFTDecoderMask',
        net_type='Basic',
        num_levels=4,
        radius=4,
        iters=12,
        corr_lookup_cfg=dict(align_corners=True),
        gru_type='SeqConv',
        act_cfg=dict(type='ReLU')),
    flow_loss_cfg=dict(
        type='SequenceLoss',
        gamma=0.8,
        loss_func_cfg=dict(
            type='RAFTLoss',
            loss_weight=1.0,
            max_flow=400.,
        )
    ),
    occlusion_loss_cfg=dict(
        type='SequenceLoss',
        gamma=0.8,
        loss_func_cfg=dict(
            type='L1Loss',
            loss_weight=100.,
        )
    ),
    freeze_bn=False,
    train_cfg=dict(),
    test_cfg=dict(
        iters=12
    ),
    init_cfg=dict(
        type='Pretrained',
        checkpoint='work_dirs/raft_8x2_100k_flyingthings3d_400x720_convertered.pth'
    )
)


steps = 100000
interval = steps//10
optimizer = dict(
    type='AdamW',
    lr=0.0004,
    betas=(0.9, 0.999),
    eps=1e-08,
    weight_decay=0.0001,
    amsgrad=False)
optimizer_config = dict(grad_clip=dict(max_norm=1.))
lr_config = dict(
    policy='OneCycle',
    max_lr=0.0004,
    total_steps=steps+100,
    pct_start=0.05,
    anneal_strategy='linear')
evaluation=dict(interval=interval, 
                metric={
                    'auc':[],
                    'add':[0.05, 0.10, 0.20, 0.50],
                    },
                save_best='average/add_10',
                rule='greater',
            )


runner = dict(type='IterBasedRunner', max_iters=steps)
num_gpus = 1
checkpoint_config = dict(interval=interval, by_epoch=False)
log_config=dict(interval=100, 
                hooks=[
                    dict(type='TextLoggerHook'),
                    dict(type='TensorboardImgLoggerHook', interval=200, image_format='HWC')
                    ])

work_dir = 'work_dirs/raft_ycbv_real'