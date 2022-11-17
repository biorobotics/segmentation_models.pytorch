import os
import cv2
import copy
import torch
import numpy as np

from torchvision import transforms

torch.seed()
np.random.seed(42)

def make_predictions(model, imagePath, labelPath, img_name, savepath, single_class, DEVICE):

    model.eval()
    with torch.no_grad():
        image = torch.Tensor(cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)/255.0).to(DEVICE)
        image = torch.unsqueeze(image, dim=0)
        image = torch.unsqueeze(image, dim=0)
        image = torch.cat([image, image, image], dim=1)

        orig = copy.deepcopy(image*255.0)
        orig = orig.cpu().numpy()[0, 0, :, :].reshape(256, 256, 1)
        orig = np.concatenate([orig, orig, orig], axis=-1).astype(int)
        if labelPath is not None:
            gtMask = cv2.imread(labelPath)

        norm = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        image = norm(image)

        predMask = np.zeros_like(orig)
        predicted_model = torch.softmax(model(image), dim=1).cpu().numpy()
        pred_mask = np.argmax(predicted_model, axis=1).squeeze(0)
        pred_mask_0 = np.where(pred_mask == 1)
        pred_mask_1 = np.where(pred_mask == 2)
        if single_class:
            for i in range(pred_mask_0[0].shape[0]):
                predMask[pred_mask_0[0][i], pred_mask_0[1][i], :] = [0, 255, 255]
            for i in range(pred_mask_1[0].shape[0]):
                predMask[pred_mask_1[0][i], pred_mask_1[1][i], :] = [0, 255, 255]
        else:
            for i in range(pred_mask_0[0].shape[0]):
                predMask[pred_mask_0[0][i], pred_mask_0[1][i], :] = [0, 255, 0]
            for i in range(pred_mask_1[0].shape[0]):
                predMask[pred_mask_1[0][i], pred_mask_1[1][i], :] = [0, 0, 255]

        gtOver = (0.75 * orig + 0.25 * gtMask).astype(int)
        predOver = (0.75 * orig + 0.25 * predMask).astype(int)
        image_concat = np.concatenate([orig,gtOver,predOver],axis=1)
        cv2.imwrite(savepath+'/'+img_name,image_concat)

        return predMask, gtMask

if __name__ == '__main__':
    model_dir = '/home/tejasr/projects/tracir_segmentation/segmentation_models.pytorch/models/'
    model = torch.load(model_dir+'/best_model.pt')

    image_path = '/home/tejasr/projects/tracir_segmentation/segmentation_models.pytorch/dataset/pig_dataset_fukuda3/test/images/pig_6_388.png'
    label_path = '/home/tejasr/projects/tracir_segmentation/segmentation_models.pytorch/dataset/pig_dataset_fukuda3/test/labels_masks/pig_6_388.png'
    save_path = '/home/tejasr/projects/tracir_segmentation/segmentation_models.pytorch/models/test'

    make_predictions(model, image_path, label_path, 'test.png', save_path, True, 'cuda')