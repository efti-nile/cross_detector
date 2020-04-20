import numpy as np
import cv2

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from progiter import ProgIter

from video_reader import VideoReader
from detectron2 import model_zoo


def get_default_predictor():
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.8  # set threshold for this model
    cfg.SOLVER.IMS_PER_BATCH = 1  # to reduce memory usage
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
    return DefaultPredictor(cfg)


class PersonDetector:

    PERSON_CID = 0

    def __init__(self, video_path, predictor_creator, decimation=None):
        assert callable(predictor_creator)
        self.video = VideoReader(video_path, decimation)
        self.predictor_creator = predictor_creator
        self.predictor = None

    def process_next_frame(self):
        retval = self.video.get_datetime_frame()
        if retval is not None:
            date, frame = retval
            outputs = self.predictor(frame)
            cid_mask = outputs['instances'].pred_classes == self.PERSON_CID
            cid_num = cid_mask.sum().item()  # total number of detections
            if cid_num:
                # copying required to detach numpy array from underlying Tensor's storage
                boxes = outputs['instances'].pred_boxes[cid_mask].tensor.cpu().numpy()
                scores = np.copy(outputs['instances'].scores[cid_mask].cpu().numpy().reshape(cid_num, 1))
                masks = outputs['instances'].pred_masks[cid_mask].cpu().numpy()
                person_masks = [np.copy(PersonDetector.extract_person_mask(m, b)) for m, b in zip(masks, boxes)]  # diff. sizes
                person_images = [np.copy(ndarr) for ndarr in PersonDetector.extract_person_images(frame, boxes)]
                boxes = [np.copy(ndarr) for ndarr in boxes]
                return date, frame, (person_masks, boxes, person_images, scores)
            else:
                return date, frame, None  # No detections
        else:
            return None  # No more frames
        
    @staticmethod
    def int_lims_from_box(box, frame_shape):
        start_x, start_y, end_x, end_y = box.astype("int")  # truncate and convert to integers
        start_x, end_x = np.clip([start_x, end_x], 0, frame_shape[1])
        start_y, end_y = np.clip([start_y, end_y], 0, frame_shape[0])
        return start_x, start_y, end_x, end_y

    @staticmethod
    def extract_person_images(frame, boxes):
        person_images = []
        for box in boxes:
            start_x, start_y, end_x, end_y = PersonDetector.int_lims_from_box(box, frame.shape)
            image = frame[start_y:end_y, start_x:end_x]
            person_images.append(image)
        return person_images
    
    @staticmethod
    def extract_person_mask(mask, box):
        start_x, start_y, end_x, end_y = PersonDetector.int_lims_from_box(box, mask.shape)
        return mask[start_y:end_y, start_x:end_x]
        

if __name__ == '__main__':
    import cv2
    pd = PersonDetector('data/test_video.mp4', get_default_predictor(), cache_dir='data')
    while True:
        retval = pd.process_next_frame()
        if retval is not None:
            date, f, detection_data = retval
            if detection_data is not None:
                m, boxes, images, scores = detection_data
                for b in boxes:
                    x1, y1, x2, y2 = b.astype("int")
                    x1, x2 = np.clip([x1, x2], 0, f.shape[1])
                    y1, y2 = np.clip([y1, y2], 0, f.shape[0])
                    cv2.rectangle(f, (x1, y1), (x2, y2), (255, 255, 0))
            cv2.imshow('W', f)
            key = cv2.waitKey(30) & 0xFF
            if key == 27:
                break
        else:
            break
