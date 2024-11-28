modality = 'b'
graph = 'nturgb+d'
work_dir = f'./work_dirs/ntu120_xset/b'

model = dict(
    type='RecognizerGCN',
    backbone=dict(
        type='ProtoGCN',
        num_prototype=100,
        tcn_ms_cfg=[(3, 1), (3, 2), (3, 3), (3, 4), ('max', 3), '1x1'],
        graph_cfg=dict(layout=graph, mode='random', num_filter=8, init_off=.04, init_std=.02)),
    cls_head=dict(type='SimpleHead', joint_cfg='nturgb+d', num_classes=120, in_channels=384, weight=0.3))

dataset_type = 'PoseDataset'
ann_file = '/data/nturgbd/ntu120_3danno.pkl'
train_pipeline = [
    dict(type='PreNormalize3D', align_spine=False),
    dict(type='RandomRot', theta=0.2),
    # Effective Augmentation
    dict(type='Spatial_Flip', dataset='nturgb+d', p=0.5),
    dict(type='GenSkeFeat', feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=100),
    dict(type='FormatGCNInput'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
val_pipeline = [
    dict(type='PreNormalize3D', align_spine=False),
    dict(type='GenSkeFeat', feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=100, num_clips=1),
    dict(type='FormatGCNInput'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
test_pipeline = [
    dict(type='PreNormalize3D', align_spine=False),
    dict(type='GenSkeFeat', feats=[modality]),
    dict(type='UniformSampleDecode', clip_len=100, num_clips=10),
    dict(type='FormatGCNInput'),
    dict(type='Collect', keys=['keypoint', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['keypoint'])
]
data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    test_dataloader=dict(videos_per_gpu=1),
    train=dict(type=dataset_type, ann_file=ann_file, pipeline=train_pipeline, split='xset_train'),
    val=dict(type=dataset_type, ann_file=ann_file, pipeline=val_pipeline, split='xset_val'),
    test=dict(type=dataset_type, ann_file=ann_file, pipeline=test_pipeline, split='xset_val'))

# setting: 4 GPU  64  0.1  ->  1 GPU  64/4=16  0.1/4=0.025
optimizer = dict(type='SGD', lr=0.025, momentum=0.9, weight_decay=0.0005, nesterov=True)
optimizer_config = dict(grad_clip=None)
lr_config = dict(policy='CosineAnnealing', min_lr=0, by_epoch=False)
total_epochs = 150
checkpoint_config = dict(interval=1)
evaluation = dict(interval=1, metrics=['top_k_accuracy'])
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
