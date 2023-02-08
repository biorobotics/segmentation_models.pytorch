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
from torch.utils.tensorboard import SummaryWriter

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
parser.add_argument('--root_data_dir', type=str, default='dataset/pig_dataset_fukuda3')
parser.add_argument('--saved_model_dir', type=str, default='models')
parser.add_argument('--model_signature', type=str, default='fukuda_pig_test') #model savefile name
parser.add_argument('--data_aug', type=bool, default=False)

# Hyperparameters
parser.add_argument('--run_mode', type=int, default=1, help='Run Mode 0 = TRAIN; Run Mode 1 = INFERENCE')
parser.add_argument('--single_class', type=bool, default=False)
parser.add_argument('--encoder_name', type=str, default='resnet34')
parser.add_argument('--encoder_weights', type=str, default='imagenet')
parser.add_argument('--preprocess', type=bool, default=False)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--metrics_threshold', type=float, default=0.5)
parser.add_argument('--num_epochs', type=int, default=100)
parser.add_argument('--early_stopping', type=bool, default=True)
parser.add_argument('--early_stopping_patience', type=int, default=10)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--ignore_index', type=int, default=-1, help='Default set to -1 (no ignore index). ')
parser.add_argument('--loss_fn', type=str, default='dice_multiclass', help='choose from dice_multiclass and bce_with_logits')
parser.add_argument('--metrics', type=str, default='iou_score', help='choose from iou_score and dice_score')
parser.add_argument('--auto_infer', type=int, default=30)
parser.add_argument('--readme', type=str, default='')
args = parser.parse_args()

root_data_dir = str(ROOT/args.root_data_dir)
saved_model_dir = str(ROOT/args.saved_model_dir)

if os.path.exists(root_data_dir) == 0:
    raise ValueError('Dataset directory not found')
if os.path.exists(saved_model_dir) == 0:
    os.mkdir(saved_model_dir)

print('\n\nTRAINING DETAILS\n')
print('Run Mode {}\nModel Signature {}\nData Augmentation {}\nLoss Function {}\nSingle Class {}\n'.format(
    args.run_mode, args.model_signature, args.data_aug, args.loss_fn, args.single_class))
print('LR {}\nNumber of Epochs {}\nEarly Stopping {}\nPatience {}\nAuto Infer {}'.format(
    args.learning_rate, args.num_epochs, args.early_stopping, args.early_stopping_patience, args.auto_infer))

model_dir = args.model_signature + NOW
try:
	os.makedirs(os.path.join(saved_model_dir, model_dir))
except:
	pass

savepath = os.path.join(saved_model_dir, model_dir)
if args.run_mode == 0:
    with open(savepath+'/readme.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)

auto_infer = args.auto_infer != 0


""" Defining Data """
if args.preprocess: preprocessing_fn = smp.encoders.get_preprocessing_fn(args.encoder_name)
else: preprocessing_fn = None

if args.probe == 'hprobe':
    train_dataset = hprobeDataset(datadir=args.root_data_dir+'/train', data_aug=args.data_aug)
    valid_dataset = hprobeDataset(datadir=args.root_data_dir+'/valid', data_aug=args.data_aug)
    test_dataset = hprobeDataset(datadir=args.root_data_dir+'/test', data_aug=args.data_aug)
elif args.probe == 'fukuda':
    train_dataset = FukudaDataset(datadir=args.root_data_dir+'/train', data_aug=args.data_aug, preproc=preprocessing_fn)
    valid_dataset = FukudaDataset(datadir=args.root_data_dir+'/valid', data_aug=args.data_aug, preproc=preprocessing_fn)
    test_dataset = FukudaDataset(datadir=args.root_data_dir+'/test', data_aug=args.data_aug, preproc=preprocessing_fn)
train_loader = DataLoader(train_dataset, shuffle=True,
    batch_size=args.batch_size)
    # num_workers=os.cpu_count())
valid_loader = DataLoader(valid_dataset, shuffle=False,
    batch_size=args.batch_size)
    # num_workers=os.cpu_count())
test_loader = DataLoader(test_dataset, shuffle=False,
    batch_size=args.batch_size)
    # num_workers=os.cpu_count())


""" Defining Model """
model = smp.Unet(
            encoder_name=args.encoder_name,
            encoder_weights=args.encoder_weights,
            in_channels=3,
            classes=3,
            ).to(DEVICE)


""" Loss, metrics and optimizer """
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

if args.metrics == 'iou_score':
    metrics = [utils.metrics.IoU(threshold=args.metrics_threshold, ignore_channels=[0], activation='softmax', single_class=args.single_class)]
elif args.metrics == 'dice_score':
    metrics = [utils.metrics.Fscore(threshold=args.metrics_threshold, ignore_channels=[0], activation='softmax', single_class=args.single_class)]
elif args.metrics == 'euclidean_dist':
    metrics = [utils.metrics.EuclideanDist(threshold=args.metrics_threshold, ignore_channels=[0], activation='softmax', single_class=args.single_class)]
elif args.metrics == 'hamming_dist':
    metrics = [utils.metrics.HammingDist(threshold=args.metrics_threshold, ignore_channels=[0], activation='softmax', single_class=args.single_class)]

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

    writer = SummaryWriter(savepath)
    print('Starting training')
    max_score = 0.0
    for i in range(args.num_epochs):
        print('\nEpoch {}'.format(i))
        train_logs = train_epoch.run(train_loader)
        writer.add_scalar('train/loss', train_logs[args.loss_fn], i)
        writer.add_scalar('train/{}'.format(args.metrics), train_logs[args.metrics], i)
        valid_logs = valid_epoch.run(valid_loader)
        writer.add_scalar('valid/loss', valid_logs[args.loss_fn], i)
        writer.add_scalar('valid/{}'.format(args.metrics), valid_logs[args.metrics], i)
        with open(savepath+'/readme.txt', 'a') as f:
            f.write('\n\nEpoch {}'.format(i))
            f.write('\n\nTrain Logs')
            json.dump(train_logs, f, indent=2)
            f.write('\n\nValid Logs')
            json.dump(valid_logs, f, indent=2)

        if max_score < valid_logs[args.metrics]:
            max_score = valid_logs[args.metrics]
            torch.save(model, savepath+'/best_model.pt',_use_new_zipfile_serialization=False)
            print('New Model saved!')
            curr_patience = 0
            print('Current patience: {}'.format(curr_patience))
        else:
            curr_patience += 1
            print('Current patience: {}'.format(curr_patience))
            if curr_patience >= args.early_stopping_patience and args.early_stopping:
                print('Early stopping activated, stopping training at epoch {}'.format(str(i)))
                break

elif args.run_mode == 1:

    print('Starting inference')
    auto_infer = False
    model = torch.load(savepath+'/best_model.pt')
    savepath = savepath+'/infer'
    if os.path.exists(savepath) == 0: os.mkdir(savepath)
    test_logs = valid_epoch.run(test_loader)
    with open(savepath+'/readme.txt', 'a') as f:
        f.write('\n\nTest Logs')
        json.dump(test_logs, f, indent=2)
    test_dir = args.root_data_dir+'/test'
    images = os.listdir(test_dir+'/images')
    count = 0
    total_score = 0.0
    for img_ in natsorted(images):
        labelPath = os.path.join(test_dir, 'labels_masks', img_)
        imagePath = os.path.join(test_dir, 'images', img_)
        pr, gt = infer.make_predictions(model, imagePath, labelPath, img_, savepath, args.single_class, DEVICE)
        y_pred = torch.Tensor(pr/255.0).permute(2, 0, 1).unsqueeze(0)
        y = torch.Tensor(gt/255.0).permute(2, 0, 1).unsqueeze(0)
        score = utils.functional.iou(y_pred, y, threshold=args.metrics_threshold, ignore_channels=[0], single_class=args.single_class)
        total_score += score
    avg_score = total_score/count
    print('Average score: {}'.format(avg_score))

elif args.run_mode == 2:

    print('Starting inference for 4 pigs')
    auto_infer = False
    model = torch.load(savepath+'/best_model.pt')
    pigs = ['pig_A', 'pig_B', 'pig_C', 'pig_D']
    dirs = ['/home/tejasr/projects/tracir_segmentation/segmentation_models.pytorch/dataset/Pig_A',
            '/home/tejasr/projects/tracir_segmentation/segmentation_models.pytorch/dataset/Pig_B',
            '/home/tejasr/projects/tracir_segmentation/segmentation_models.pytorch/dataset/Pig_C',
            '/home/tejasr/projects/tracir_segmentation/segmentation_models.pytorch/dataset/Pig_D']
    for pig, test_dir in zip(pigs, dirs):
        savepath1 = savepath+'/infer_'+pig
        if os.path.exists(savepath1) == 0: os.mkdir(savepath1)
        print(pig)
        test_dataset = FukudaDataset(datadir=test_dir, data_aug=args.data_aug, preproc=preprocessing_fn)
        test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch_size)
        test_logs = valid_epoch.run(test_loader)
        with open(savepath1+'/readme.txt', 'a') as f:
            f.write('\n\nTest Logs')
            json.dump(test_logs, f, indent=2)
        images = os.listdir(test_dir+'/images')
        count = 0
        total_score = 0.0
        # import ipdb; ipdb.set_trace()
        for img_ in natsorted(images):
            count += 1
            labelPath = os.path.join(test_dir, 'labels_masks', img_)
            imagePath = os.path.join(test_dir, 'images', img_)
            pr, gt = infer.make_predictions(model, imagePath, labelPath, img_, savepath1, args.single_class, DEVICE)
            y_pred = torch.Tensor(pr/255.0).permute(2, 0, 1).unsqueeze(0)
            y = torch.Tensor(gt/255.0).permute(2, 0, 1).unsqueeze(0)
            score = utils.functional.iou(y_pred, y, threshold=args.metrics_threshold, ignore_channels=[0], single_class=args.single_class)
            total_score += score
        avg_score = total_score/count
        print('Average score: {}'.format(avg_score))

""" Auto Infer """
if auto_infer:
    print('Starting auto-infer')
    model = torch.load(savepath+'/best_model.pt')
    savepath = savepath+'/auto_infer'
    if os.path.exists(savepath) == 0: os.mkdir(savepath)
    test_logs = valid_epoch.run(test_loader)
    with open(savepath+'/readme.txt', 'a') as f:
        f.write('\nAuto Infer: Test Logs')
        json.dump(test_logs, f, indent=2)
    test_dir = args.root_data_dir+'/test'
    images = os.listdir(test_dir+'/images')
    count = 0
    total_score = 0.0
    for im in range(args.auto_infer):
        count += 1
        img_ = images[im]
        labelPath = os.path.join(test_dir, 'labels_masks', img_)
        imagePath = os.path.join(test_dir, 'images', img_)
        pr, gt = infer.make_predictions(model, imagePath, labelPath, img_, savepath, args.single_class, DEVICE)
        y_pred = torch.Tensor(pr/255.0).permute(2, 0, 1).unsqueeze(0)
        y = torch.Tensor(gt/255.0).permute(2, 0, 1).unsqueeze(0)
        score = utils.functional.iou(y_pred, y, threshold=args.metrics_threshold, ignore_channels=[0], single_class=args.single_class)
        total_score += score
    avg_score = total_score/count
    print('Average score: {}'.format(avg_score))