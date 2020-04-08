import numpy as np
import cv2
import torch
import torchreid

from torchvision.transforms import ToTensor, Normalize


class ReID:

    def __init__(self, model_name, checkpoint_path, num_classes):
        self.model = torchreid.models.build_model(model_name, num_classes)
        self.model = self.model.cuda()
        torchreid.utils.resume_from_checkpoint(checkpoint_path, self.model)
        self.model.eval()

    @torch.no_grad()
    def get_embedding(self, img):
        emb = self.model(ReID.img2tensor(img).cuda())
        return np.array(emb.cpu())  # return embedding vector as 1x512 2d numpy array

    def get_embeddings_by_bboxes(self, img, bboxes, masks=None):
        """
        Calculate embeddings for given image regions. A mask can be used to
        remove background. Background pixels are replaced by mean value for
        a region.

        :param img: numpy array HxWx3
        :param bboxes: numpy array Nx4 with rows like [xA, yA, xB, yB]
        :param mask: numpy boolean array HxW

        :returns: numpy array (number of bboxes)x(length of emb. vector)
        """
        embeddings = np.array([])

        if masks is None:
            for xA, yA, xB, yB in bboxes.astype("int"):
                roi = img[yA:yB, xA:xB]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                emb = self.get_embedding(roi)
                embeddings = np.vstack([embeddings, emb]) if embeddings.size else emb
        else:
            for (xA, yA, xB, yB), mask in zip(bboxes.astype("int"), masks):
                roi = img[yA:yB, xA:xB]
                roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

                mask_3ch = np.repeat(np.expand_dims(mask, 2), 3, axis=2)[yA:yB, xA:xB]

                mean = roi.mean(axis=0).mean(axis=0)  # a mean pixel
                tmp = np.tile(mean, (roi.shape[0], roi.shape[1], 1))

                tmp[mask_3ch] = roi[mask_3ch]

                emb = self.get_embedding(tmp)
                embeddings = np.vstack([embeddings, emb]) if embeddings.size else emb

        return embeddings

    @staticmethod
    def img2tensor(img):
        img = cv2.resize(img, (128, 256))
        tensor = (ToTensor())(img)
        tensor = (Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))(tensor)  # ResNet
        tensor = tensor.float()
        return tensor.view(1, 3, 256, 128)
