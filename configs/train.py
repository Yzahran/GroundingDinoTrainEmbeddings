data_root = '/GDDemoTokens/Dataset/singleclevr_rubber/'
_base_ = '/GDDemoTokens/configs/custom_gd.py'


randomness = dict(
    seed = 42,
    deterministic=True
)
class_name =    (
    "blue cube",
    "red cube",
    "green sphere",
    "purple sphere",
    "brown cylinder",
    "yellow cylinder",
    "green cube",
    "purple cube",
    "yellow cube",
    "brown cube",
    "red sphere",
    "blue sphere",
    "yellow sphere",
    "brown sphere",
    "red cylinder",
    "blue cylinder",
    "green cylinder",
    "purple cylinder",
    ) 

metainfo  =dict(classes=class_name)

settings_training = dict(train=True)
model = dict(bbox_head=dict(num_classes=len(class_name),train_settings=settings_training),
            train_settings=settings_training,
            language_model=dict(train_settings=settings_training)
)
train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/10shot6comp.json',
        data_prefix=dict(img='images/')))


val_dataloader = dict(

    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/10shot6comp.json',
        data_prefix=dict(img='images/')))

test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/10shot6comp.json')

test_evaluator = val_evaluator

max_epoch = 20

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=100)

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=30),
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epoch,
        by_epoch=True,
        milestones=[15],
        gamma=0.1)
]


optim_wrapper = dict(
    optimizer=dict(lr=0.0),
    paramwise_cfg=dict(
        custom_keys={
            'language_model.language_backbone.body.model.embeddings.word_embeddings.weight':dict(lr=5e-05 ),
        }))
auto_scale_lr = dict(base_batch_size=2)
