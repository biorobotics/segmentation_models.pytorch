import os
import torch
import numpy as np 
import cv2
import copy
import ipdb

def make_predictions(model, imagePath, labelPath, img_name, savepath, DEVICE):

    savepath = savepath+'/infer'
    if os.path.exists(savepath) == 0: os.mkdir(savepath)
    model.eval()

    with torch.no_grad():
        image = torch.Tensor(cv2.imread(imagePath, cv2.IMREAD_GRAYSCALE)).to(DEVICE)
        image = torch.unsqueeze(image, dim = 0)
        image = torch.unsqueeze(image, dim = 0)
        image = torch.cat([image,image,image],dim=1)

        orig = copy.deepcopy(image)
        orig = orig.cpu().numpy()[0,0,:,:].reshape(256,256,1)
        orig = np.concatenate([orig,orig,orig],axis=-1).astype(int)
        if labelPath is not None:
            gtMask = cv2.imread(labelPath)

        predMask = np.zeros_like(orig)
        predicted_model = torch.softmax(model(image),dim=1).cpu().numpy()
        pred_mask = np.argmax(predicted_model,axis=1).squeeze(0)
        pred_mask_0 = np.where(pred_mask == 1)
        pred_mask_1 = np.where(pred_mask == 2)
        for i in range(pred_mask_0[0].shape[0]):
            predMask[pred_mask_0[0][i],pred_mask_0[1][i],:] = [0,255,0]
        for i in range(pred_mask_1[0].shape[0]):
            predMask[pred_mask_1[0][i],pred_mask_1[1][i],:] = [0,0,255]
        # ipdb.set_trace()


        # sigmoid_model = torch.softmax(model(image)).cpu().numpy()
        # predMask = np.ones_like(sigmoid_model)*0
        # for i in range(sigmoid_model.shape[1]):
        #     loc = np.where(sigmoid_model[0,i,:,:]>0.75)
        #     for idx in range(loc[0].shape[0]):
        #         predMask[0,i,loc[0][idx],loc[1][idx]] = 255


        # predMask = np.transpose(predMask[0,:,:,:], (1, 2, 0)).astype(int)
        if labelPath is None:
            gtMask = (0.75 * orig + 0.25 * predMask).astype(int)
        image_concat = np.concatenate([orig,gtMask,predMask],axis=1)
        cv2.imwrite(savepath+'/'+img_name,image_concat)