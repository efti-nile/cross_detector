import os
import math

import numpy as np
import cv2

from cached_person_detector import CachedPersonDetector
from person_detector import PersonDetector, get_default_predictor

# TODO: I think it's better to track upper point of bbox
# NOTE: What if multiple gates on a video?
# NOTE: Porojecting an anchor onto the gate line to enable vertical gates


def get_anchor(box):
    """Box's in (X1,Y1,X2,Y2) form"""
    anchor_x = .5 * (box[0] + box[2])  # center
    anchor_y = min(box[1], box[3])  # upper point
    return anchor_x, anchor_y


class CrossDetector:

    def __init__(self, person_detector, gate_pt1, gate_pt2, gate_width=50, hysteresis=25):
        """
        Args:
            person_detector:
            gate_pt1: left point of the gate
            gate_pt2: right point of the gate
            gate_width:
            hysteresis:
        """
        assert len(gate_pt1) == 2 and len(gate_pt2) == 2
        self.gate_pt1, self.gate_pt2 = gate_pt1, gate_pt2
        self.person_detector = person_detector
        self.gate_width = gate_width
        self.track_boxes = []
        self.hysteresis = hysteresis
        self.eof = False

    def process_next_frame(self):
        """
        Returns:
            A list of `TrackingBox` instances containing person tracks

        Raises:
            EOFError: the end of video
        """
        out = []
        retval = self.person_detector.process_next_frame()

        if retval is not None:
            date, frame, detection_data = retval

            if detection_data is not None:
                masks, boxes, images, _ = self.apply_x_limits(detection_data)

                for tb in self.track_boxes:
                    masks, boxes, images = tb.update(masks, boxes, images, date)

                    d = self.distance_to_point(get_anchor(tb.boxes[-1]))
                    if d > self.gate_width / 2 + self.hysteresis:
                        out.append(tb)  # return a complete `TrackingBox`
                        self.track_boxes.remove(tb)

                self.track_boxes = [tb for tb in self.track_boxes if tb.ttl > 0]

                for mask, box, img in zip(masks, boxes, images):
                    d = self.distance_to_point(get_anchor(box))
                    if d < self.gate_width / 2 - self.hysteresis:
                        self.track_boxes.append(TrackingBox(mask, box, img, date))

        return out

    def apply_x_limits(self, detection_data):
        """Removes detections out of x limits"""
        retval = []
        x_min = min(self.gate_pt1[0], self.gate_pt2[0])
        x_max = max(self.gate_pt1[0], self.gate_pt2[0])
        for detection in detection_data:
            _, box, _, _ = detection
            x1, _, x2, _ = box
            if x_min <= min(x1, x2) and max(x1, x2) <= x_max:
                retval.append(detection)
        return retval

    def distance_to_point(self, pt):
        # distance formula https://mathworld.wolfram.com/Point-LineDistance2-Dimensional.html
        x0, y0 = pt
        x1, y1 = self.gate_pt1
        x2, y2 = self.gate_pt2
        return abs((x2-x1)*(y1-y0) - (x1-x0)*(y2-y1)) / math.sqrt((x2-x1)**2 + (y2-y1)**2)

    def end_of_video(self):
        return self.person_detector.video.no_more_frames


def print_mask(mask):
    print('\n'.join([''.join([str(int(el)) for el in row]) for row in mask]))


class TrackingBox:

    IOU_THRESH = 0.3
    TTL = 3
    DATA_LISTS = ('masks', 'boxes', 'imgs', 'dates')

    def __init__(self, init_mask, init_box, init_img, init_date):
        assert isinstance(init_box, np.ndarray) and init_box.ndim == 1 and init_box.size == 4
        self.masks = [init_mask]
        self.boxes = [init_box]
        self.imgs = [init_img]
        self.dates = [init_date]
        self.track_len = 1
        self.ttl = self.TTL
        self.reid_id = None

    def IoU_with_mask(self, mask, box):
        assert isinstance(box, np.ndarray) and box.ndim == 1 and box.size == 4
        if self.is_intersection(box):
            sbox = self.boxes[-1].copy()  # self box
            smask = self.masks[-1]  # self mask
            cbox = box.copy()  # copy of box

            # Translate detections to the origin to reduce an amount of padding
            min_x = min(cbox[0], sbox[0])
            min_y = min(cbox[1], sbox[1])
            offset = np.array([min_x, min_y, min_x, min_y])
            cbox = np.round(cbox - offset).astype(np.int)
            sbox = np.round(sbox - offset).astype(np.int)

            # Padding to common size
            total_width_px = max(cbox[2], sbox[2]) + 1  # +1 just in case
            total_height_px = max(cbox[3], sbox[3]) + 1  # +1 just in case
            smask_with_padding = np.full((total_height_px, total_width_px), False)
            mask_with_padding = np.full((total_height_px, total_width_px), False)
            smask_with_padding[sbox[1]:(sbox[1] + smask.shape[0]), sbox[0]:(sbox[0] + smask.shape[1])] = smask
            mask_with_padding[cbox[1]:(cbox[1] + mask.shape[0]), cbox[0]:(cbox[0] + mask.shape[1])] = mask

            # IoU
            iou = (smask_with_padding & mask_with_padding).sum() / (smask_with_padding | mask_with_padding).sum()
            return iou
        else:
            return 0

    def is_intersection(self, box):
        x1, y1, x2, y2 = box
        x3, y3, x4, y4 = self.boxes[-1]
        x_cross = TrackingBox.intersection_1d(x1, x2, x3, x4)
        y_cross = TrackingBox.intersection_1d(y1, y2, y3, y4)
        return x_cross * y_cross > 0

    @staticmethod
    def intersection_1d(start1, end1, start2, end2):
        assert start1 < end1 and start2 < end2
        start = max(start1, start2)
        end = min(end1, end2)
        if start < end:
            return end - start
        else:
            return 0

    def update(self, masks_in_frame, boxes_in_frame, imgs_in_frame, date):
        assert len(masks_in_frame) == len(boxes_in_frame) == len(imgs_in_frame)
        assert isinstance(masks_in_frame, list) and isinstance(boxes_in_frame, list) \
            and isinstance(imgs_in_frame, list)

        ious = np.array([self.IoU_with_mask(m, b) for m, b in zip(masks_in_frame, boxes_in_frame)])
        max_iou_idx = np.argmax(ious)
        max_iou = ious[max_iou_idx]
        self.ttl -= 1

        if max_iou > self.IOU_THRESH:
            # Append a new detection to the track
            self.masks.append(masks_in_frame.pop(max_iou_idx))
            self.boxes.append(boxes_in_frame.pop(max_iou_idx))
            self.imgs.append(imgs_in_frame.pop(max_iou_idx))
            self.dates.append(date)
            self.track_len += 1

            self.ttl = self.TTL  # reset ttl

        return masks_in_frame, boxes_in_frame, imgs_in_frame

    def _write_images(self):
        dir_name = str(id(self))
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        for idx, img in enumerate(self.imgs):
            cv2.imwrite(os.path.join(dir_name, f'{idx}.jpg'), img)


if __name__ == '__main__':
    cd = CrossDetector(CachedPersonDetector('data/test_video.mp4', get_default_predictor(), cache_dir='data'), 800)
    tbs = []
    while True:
        tbs += cd.process_next_frame()
        if cd.end_of_video():
            break

    for t in tbs:
        t._write_images()
