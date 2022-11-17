import torch
from scipy.spatial import distance


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [channel for channel in range(xs[0].shape[1]) if channel not in ignore_channels]
        xs = [torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device)) for x in xs]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def _get_single_class(pr, gt):
    pr_sum = torch.sum(pr, dim=1).clamp(min=0.0, max=1.0)
    gt_sum = torch.sum(gt, dim=1).clamp(min=0.0, max=1.0)
    return pr_sum, gt_sum


def iou(pr, gt, eps=1e-7, threshold=0.5, ignore_channels=None, single_class=False):
    """Calculate Intersection over Union between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    if single_class: pr, gt = _get_single_class(pr, gt)

    intersection = torch.sum(gt * pr)
    union = torch.sum(gt) + torch.sum(pr) - intersection + eps

    score = (intersection + eps) / union

    return score.cpu().detach().numpy()


jaccard = iou


def f_score(pr, gt, beta=1, eps=1e-7, threshold=0.5, ignore_channels=None, single_class=False):
    """Calculate F-score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        beta (float): positive constant
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: F score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    if single_class: pr, gt = _get_single_class(pr, gt)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp
    fn = torch.sum(gt) - tp

    score = ((1 + beta**2) * tp + eps) / ((1 + beta**2) * tp + beta**2 * fn + fp + eps)

    return score.cpu().detach().numpy()


def accuracy(pr, gt, threshold=0.5, ignore_channels=None, single_class=False):
    """Calculate accuracy score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """
    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    if single_class: pr, gt = _get_single_class(pr, gt)

    tp = torch.sum(gt == pr, dtype=pr.dtype)
    score = tp / gt.view(-1).shape[0]

    return score.cpu().detach().numpy()


def precision(pr, gt, eps=1e-7, threshold=0.5, ignore_channels=None, single_class=False):
    """Calculate precision score between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: precision score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    if single_class: pr, gt = _get_single_class(pr, gt)

    tp = torch.sum(gt * pr)
    fp = torch.sum(pr) - tp

    score = (tp + eps) / (tp + fp + eps)

    return score.cpu().detach().numpy()


def recall(pr, gt, eps=1e-7, threshold=0.5, ignore_channels=None, single_class=False):
    """Calculate Recall between ground truth and prediction
    Args:
        pr (torch.Tensor): A list of predicted elements
        gt (torch.Tensor):  A list of elements that are to be predicted
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: recall score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    if single_class: pr, gt = _get_single_class(pr, gt)

    tp = torch.sum(gt * pr)
    fn = torch.sum(gt) - tp

    score = (tp + eps) / (tp + fn + eps)

    return score.cpu().detach().numpy()

def euclidean_distance(pr, gt, eps=1e-7, threshold=0.5, ignore_channels=None, single_class=False):
    """Calculate the euclidean distance between the centers of the ground truth and prediction contours
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: euclidean distance
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    if single_class: pr, gt = _get_single_class(pr, gt)

    pr_f = torch.flatten(pr).cpu().detach().numpy()
    gt_f = torch.flatten(gt).cpu().detach().numpy()

    dist = distance.euclidean(gt_f, pr_f)

    return dist

def hamming_distance(pr, gt, eps=1e-7, threshold=0.5, ignore_channels=None, single_class=False):
    """Calculate the hamming distance between the centers of the ground truth and prediction contours
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: hamming distance
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    if single_class: pr, gt = _get_single_class(pr, gt)

    pr_f = torch.flatten(pr).cpu().detach().numpy()
    gt_f = torch.flatten(gt).cpu().detach().numpy()

    dist = distance.euclidean(gt_f, pr_f)

    return dist


if __name__ == '__main__':
    import cv2
    import matplotlib.pyplot as plt

    image_path = '/home/tejasr/projects/tracir_segmentation/scratchpad/data/dummy_data2.png'
    image = torch.Tensor(cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)/255.0)
    label = torch.Tensor(cv2.imread(image_path))

    image = torch.unsqueeze(image, dim = 0)

    label_background = torch.ones_like(label[:,:,0])*255
    label_background = label_background - label[:,:,1] - label[:,:,2]

    label[:,:,0] = label_background

    image = torch.stack([image, image, image], dim=0).squeeze(1)
    # if self.preprocessing:
    #     image = self.preprocessing(image.permute(1,2,0))
    #     image = image.permute(2,0,1).type(torch.FloatTensor)
    label = label.permute((2, 0, 1))/255.0
    label = (label>0).float()

    image = torch.unsqueeze(image, dim = 0)
    label = torch.unsqueeze(label, dim = 0)

    print('IoU: ', iou(label, label))
    print('IoU (ignore channels): ', iou(label, label, ignore_channels=[0]))
    print('IoU (single class): ', iou(label, label, single_class=True))
    print('IoU (ignore channels, single class): ', iou(label, label, ignore_channels=[0], single_class=True))