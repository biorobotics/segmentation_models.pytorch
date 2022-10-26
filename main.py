import json
import torch
import os, sys
import argparse
import numpy as np
import segmentation_models_pytorch as smp
import segmentation_models_pytorch.utils as utils

from pathlib import Path
from datetime import datetime
from natsort import natsorted
from torch.utils.data import DataLoader

import infer
from dataset import hprobeDataset
from dataset import FukudaDataset

import ipdb

torch.seed()
np.random.seed(42)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NOW = datetime.today().strftime('_%Y-%m-%d')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


""" Define training args """
parser = argparse.ArgumentParser(description='Training')
# Study type and data paths
parser.add_argument('--probe', type=str, default='fukuda')
parser.add_argument('--study_type', type=str, default='pig')
parser.add_argument('--root_data_dir', type=str, default=str(ROOT/'dataset/pig_dataset_fukuda'))
parser.add_argument('--saved_model_dir', type=str, default=str(ROOT/'models'))
parser.add_argument('--model_signature', type=str, default='fukuda_pig_test') #model savefile name
parser.add_argument('--data_aug', type=bool, default=False)

# Hyperparameters
parser.add_argument('--run_mode', type=int, default=1, help='Run Mode 0 = TRAIN; Run Mode 1 = INFERENCE')
parser.add_argument('--encoder_name', type=str, default='resnet34')
parser.add_argument('--encoder_weights', type=str, default='imagenet')
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--IoU_threshold', type=float, default=0.5)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--early_stopping', type=bool, default=True)
parser.add_argument('--early_stopping_patience', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--ignore_index', type=int, default=-1, help='Default set to -1 (no ignore index). ')
parser.add_argument('--loss_fn', type=str, default='dice_multiclass') #choose from dice_multiclass and bce_with_logits
parser.add_argument('--auto_infer', type=int, default=30)
parser.add_argument('--readme', type=str, default='')
args = parser.parse_args()

if os.path.exists(args.root_data_dir) == 0:
    raise ValueError('Dataset directory not found')
if os.path.exists(args.saved_model_dir) == 0:
    os.mkdir(args.saved_model_dir)

print('\n\nTRAINING DETAILS\n')
print('Run Mode {}\nModel Signature {}\nData Augmentation {}\nLoss Function {}\n'.format(
    args.run_mode, args.model_signature, args.data_aug, args.loss_fn))
print('LR {}\nNumber of Epochs {}\nEarly Stopping {}\nPatience {}\nAuto Infer {}'.format(
    args.learning_rate, args.num_epochs, args.early_stopping, args.early_stopping_patience, args.auto_infer))

model_dir = args.model_signature
try:
	os.makedirs(os.path.join(args.saved_model_dir, model_dir))
except:
	pass

savepath = os.path.join(args.saved_model_dir, model_dir)
with open(savepath+'/readme.txt', 'w') as f:
    json.dump(args.__dict__, f, indent=2)

auto_infer = args.auto_infer != 0


""" Defining Data """
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
    # num_workers=os.cpu_count())s

len_train_data = len(train_dataset)
len_valid_data = len(valid_dataset)


""" Defining Model """
if args.loss_fn == "dice_multiclass":
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
if args.ignore_index < 0:
    if args.loss_fn == 'bce_with_logits':
        loss = smp.losses.SoftBCEWithLogitsLoss()
    elif args.loss_fn == 'dice_multiclass':
        loss = smp.losses.DiceLoss(mode="multiclass")
else:
    if args.loss_fn == 'bce_with_logits':
        loss = smp.losses.SoftBCEWithLogitsLoss(ignore_index=args.ignore_index)
    elif args.loss_fn == 'dice_multiclass':
        loss = smp.losses.DiceLoss(mode="multiclass", ignore_index=args.ignore_index)

loss.__name__ = args.loss_fn
metrics = [utils.metrics.IoU(threshold=args.IoU_threshold, ignore_channels=[0], activation='softmax')]
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

if args.run_mode == 0:

    print('Starting training')
    max_score = 0.0
    for i in range(args.num_epochs):
        print('\nEpoch {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        valid_logs = valid_epoch.run(valid_loader)

        if max_score < valid_logs['iou_score']:
            max_score = valid_logs['iou_score']
            torch.save(model, savepath+'/best_model.pt',_use_new_zipfile_serialization=False)
            print('New Model saved!')
            curr_patience = 0
        else:
            curr_patience += 1
            if curr_patience >= args.early_stopping_patience and args.early_stopping:
                print('Early stopping activated, stopping training at epoch {}'.format(str(i)))
                break

elif args.run_mode == 1:

    print('Starting inference')
    auto_infer = False
    savepath = savepath+'/infer'
    if os.path.exists(savepath) == 0: os.mkdir(savepath)
    model = torch.load(args.saved_model_dir+'/'+args.model_signature+'best_model.pt')
    test_logs = valid_epoch.run(test_loader)
    test_dir = args.root_data_dir+'/test'
    images = os.listdir(test_dir+'/images')
    for img_ in natsorted(images):
        labelPath = os.path.join(test_dir, 'labels', img_)
        imagePath = os.path.join(test_dir, 'images', img_)
        infer.make_predictions(model, imagePath, labelPath, img_, savepath, DEVICE)


""" Auto Infer """
if auto_infer:
    print('Starting auto-infer')
    model = torch.load(savepath+'/best_model.pt')
    savepath = savepath+'/auto_infer'
    if os.path.exists(savepath) == 0: os.mkdir(savepath)
    test_logs = valid_epoch.run(test_loader)
    test_dir = args.root_data_dir+'/test'
    images = os.listdir(test_dir+'/images')
    for im in range(args.auto_infer):
        img_ = images[im]
        labelPath = os.path.join(test_dir, 'labels', img_)
        imagePath = os.path.join(test_dir, 'images', img_)
        infer.make_predictions(model, imagePath, labelPath, img_, savepath, DEVICE)