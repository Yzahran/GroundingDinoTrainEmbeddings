_base_  = '/GroundingDinoTrainEmbeddings/configs/custom_gd.py'

data_root = '/GroundingDinoTrainEmbeddings/Dataset/singleclevr_rubber/'
class_name =   ("red cube",
      "blue cube",
      "green cube",
      "purple cube",
      "yellow cube",
      "brown cube",
      "red sphere",
      "blue sphere",
      "green sphere",
      "purple sphere",
      "yellow sphere",
      "brown sphere",
      "red cylinder",
      "blue cylinder",
      "green cylinder",
      "purple cylinder",
      "yellow cylinder",
      "brown cylinder")

num_classes = len(class_name)
palettes = [
    (220, 20, 60), (255, 0, 0), (0, 255, 0), (0, 0, 255),
    (255, 255, 0), (0, 255, 255), (128, 128, 128), (165, 42, 42),
    (255, 192, 203), (128, 0, 128), (255, 165, 0), (0, 0, 128),
    (0, 128, 0), (255, 255, 255), (192, 192, 192), (139, 69, 19),
    (255, 69, 0), (218, 112, 214)
]
metainfo  =dict(classes=class_name,palette=palettes,
            
)
settings_training = dict(train=False)
model = dict(bbox_head=dict(num_classes=num_classes),
            train_settings=settings_training,
            language_model=dict(train_settings=settings_training)
)

train_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file='annotations/test.json',  
        data_prefix=dict(img='images/')))

val_dataloader = dict(
    dataset=dict(
        metainfo=metainfo,
        data_root=data_root,
        ann_file='annotations/test.json',
        data_prefix=dict(img='images/')))


test_dataloader = val_dataloader

val_evaluator = dict(ann_file=data_root + 'annotations/test.json')
#uncomment this if you want to evaluate with NMS mAP
#val_evaluator = dict(ann_file=data_root + 'annotations/10shot18comp.json',type='NMSCocoMetric')
test_evaluator = val_evaluator

max_epoch = 12

default_hooks = dict(
    checkpoint=dict(interval=1, max_keep_ckpts=1, save_best='auto'),
    logger=dict(type='LoggerHook', interval=5))
train_cfg = dict(max_epochs=max_epoch, val_interval=1)

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
    optimizer=dict(lr=0.00005),
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'backbone': dict(lr_mult=0.1),
            'language_model': dict(lr_mult=0),
        }))

auto_scale_lr = dict(base_batch_size=2)
