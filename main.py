import os
import json
import argparse
import ipdb
import random
import numpy as np 
from datetime import datetime
from natsort import natsorted

import infer
from dataset import hprobeDataset
from dataset import FukudaDataset
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as utils

import torch
from torch.utils.data import DataLoader


random.seed(0)
np.random.seed(42)
torch.seed()
# torch.use_deterministic_algorithms(True)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NOW = datetime.today().strftime('_%Y-%m-%d')

'''
Define training args
'''
parser = argparse.ArgumentParser(description='Training')
# Study type and data paths
parser.add_argument('--probe', type=str, default='fukuda')
parser.add_argument('--study_type', type=str, default='pig')
parser.add_argument('--root_data_dir', type=str, default='/home/abhimanyu8713/abhimanyu_research/roboTRAC_research/segmentation/data/pig_dataset')
parser.add_argument('--saved_model_dir', type=str, default='/home/abhimanyu8713/abhimanyu_research/roboTRAC_research/segmentation/tracir_segmentation/models')
parser.add_argument('--model_signature', type=str, default='fukuda_pig_test4') #model savefile name
parser.add_argument('--data_aug', type=bool, default=False)

# Hyperparameters
parser.add_argument('--encoder_name', type=str, default='resnet34')
parser.add_argument('--encoder_weights', type=str, default='imagenet')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--IoU_threshold', type=float, default=0.5)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--early_stopping', type=bool, default=True)
parser.add_argument('--early_stopping_patience', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--ignore_index', type=int, default=0)
parser.add_argument('--loss_fn', type=str, default='dice_multiclass') #choose from dice_binary, dice_multiclass and bce_with_logits
parser.add_argument('--smp_training', type=bool, default=True)
parser.add_argument('--readme', type=str, default='')
parser.add_argument('--auto_infer', type=bool, default=False)
parser.add_argument('--run_mode', type=int, default=1)
args = parser.parse_args()

#check args
if os.path.exists(args.root_data_dir) == 0:
    raise ValueError('Root dir not found')
if os.path.exists(args.saved_model_dir) == 0:
    os.mkdir(args.saved_model_dir)

###TODO####
print('\n\nTRAINING DETAILS')
print('LR {}\nNumber of Epochs {}\nEarly Stopping {}\nPatience {}\n'.format(args.learning_rate, args.num_epochs, args.early_stopping, args.early_stopping_patience))

model_dir = args.model_signature+NOW
try:
	os.makedirs(os.path.join(args.saved_model_dir, model_dir))
except:
	pass

savepath = os.path.join(args.saved_model_dir, model_dir)
with open(savepath+'/readme.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

##defining data##
if args.probe == 'hprobe':
    train_dataset = hprobeDataset(datadir=args.root_data_dir+'/train', data_aug=args.data_aug)
    valid_dataset = hprobeDataset(datadir=args.root_data_dir+'/valid', data_aug=args.data_aug)
    test_dataset = hprobeDataset(datadir=args.root_data_dir+'/test', data_aug=args.data_aug)
elif args.probe == 'fukuda':
    train_dataset = FukudaDataset(datadir=args.root_data_dir+'/train', data_aug=args.data_aug)
    valid_dataset = FukudaDataset(datadir=args.root_data_dir+'/valid', data_aug=args.data_aug)
    test_dataset = FukudaDataset(datadir=args.root_data_dir+'/test', data_aug=args.data_aug)
train_loader = DataLoader(train_dataset, shuffle=True,
    batch_size=args.batch_size) 
    # num_workers=os.cpu_count())
valid_loader = DataLoader(valid_dataset, shuffle=False,
    batch_size=args.batch_size)
    # num_workers=os.cpu_count())
test_loader = DataLoader(test_dataset, shuffle=False,
    batch_size=args.batch_size)
    # num_workers=os.cpu_count())


len_train_data = len(train_dataset)
len_valid_data = len(valid_dataset)

##defining model##

if args.loss_fn == "dice_multiclass" or args.loss_fn == "dice_binary":
    model = smp.Unet(
                encoder_name=args.encoder_name,
                encoder_weights=args.encoder_weights,
                in_channels=3,
                classes=3,
                ).to(DEVICE)
else:
    model = smp.Unet(
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
            in_channels=3,
            classes=3,
            ).to(DEVICE)
preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder_name)


""" Loss and optimizer """
if args.loss_fn == 'bce_with_logits':
    loss = smp.losses.SoftBCEWithLogitsLoss(ignore_index=0)
elif args.loss_fn == 'dice_multiclass':
    loss = smp.losses.DiceLoss(mode="multiclass",ignore_index=0)

loss.__name__ = args.loss_fn
metrics = [utils.metrics.IoU(threshold=args.IoU_threshold,ignore_channels=[0], activation='softmax')]
optimizer = torch.optim.Adam([dict(params=model.parameters(), lr=args.learning_rate)])

""" Training """
train_epoch = utils.train.TrainEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    optimizer=optimizer,
    device=DEVICE,
    verbose=True
)
valid_epoch = utils.train.ValidEpoch(
    model, 
    loss=loss, 
    metrics=metrics, 
    device=DEVICE,
    verbose=True
)
# ipdb.set_trace()

if args.run_mode == 0:
    max_score = 0.0
    for i in range(args.num_epochs):
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, savepath+'/best_model.pth',_use_new_zipfile_serialization=False)
            print('New Model saved!')


elif args.run_mode == 1:

    # ipdb.set_trace()
    model = torch.load(args.saved_model_dir+'/'+args.model_signature+'_2022-10-20/'+'best_model.pth')
    test_logs = valid_epoch.run(test_loader)
    test_dir = args.root_data_dir+'/test'
    images = os.listdir(test_dir+'/images')
    for img_ in natsorted(images)[0:]:
        labelPath = os.path.join(test_dir, 'labels', img_)
        imagePath = os.path.join(test_dir, 'images', img_)
        infer.make_predictions(model, imagePath, labelPath, img_,savepath, DEVICE)


# if args.auto_infer:
# 	# TODO: repair this
# 	savepath = savepath+'/infer'
# 	if os.path.exists(savepath) == 0: os.mkdir(savepath)
# 	infer = Infer(args, savepath)
# 	infer.run()

