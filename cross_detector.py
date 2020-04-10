import numpy as np
import cv2
import os

from cached_person_detector import CachedPersonDetector
from person_detector import PersonDetector, get_default_predictor


class CrossDetector:

    def __init__(self, person_detector, gate_y_coor, gate_width=50, hysteresis=25):
        self.person_detector = person_detector
        self.gate_y_coor = gate_y_coor
        self.gate_width = gate_width
        self.track_boxes = []
        self.hysteresis = hysteresis

    def process_next_frame(self):
        """
        Returns:
            A list of `TrackingBox` instances containing person tracks

        Raises:
            EOFError: the end of video
        """
        out = []
        date, frame, detection_data = self.person_detector.process_next_frame()
        if detection_data is not None:
            masks, boxes, images, _ = detection_data
            for tb in self.track_boxes:
                masks, boxes, images = tb.update(masks, boxes, images, date)
                if tb.lower_y > (self.gate_y_coor + self.gate_width // 2 + self.hysteresis) \
                        or tb.lower_y < (self.gate_y_coor - self.gate_width // 2 - self.hysteresis):
                    out.append(tb)  # return a complete `TrackingBox`
                    self.track_boxes.remove(tb)
            for mask, box, img in zip(masks, boxes, images):
                _, y1, _, y2 = box
                lower_y = max(y1, y2)
                if (self.gate_y_coor - self.gate_width // 2) < lower_y < (self.gate_y_coor + self.gate_width // 2):
                    self.track_boxes.append(TrackingBox(mask, box, img, date))
        return out


class TrackingBox:

    IOU_THRESH = 0.5

    def __init__(self, init_mask, init_box, init_img, init_date):
        assert isinstance(init_box, np.ndarray) and init_box.ndim == 1 and init_box.size == 4
        self.masks = [init_mask]
        self.boxes = [init_box]
        self.imgs = [init_img]
        self.dates = [init_date]
        self.track_len = 1
        self.lower_y = max(init_box[1], init_box[3])  # the lower point has maximum y coordinate

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
            return (smask_with_padding & mask_with_padding).sum() / (smask_with_padding | mask_with_padding).sum()
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

        for i in range(len(masks_in_frame)):
            if self.IoU_with_mask(masks_in_frame[i], boxes_in_frame[i]) > self.IOU_THRESH:
                # Append a new detection to the track
                self.masks.append(masks_in_frame.pop(i))
                self.boxes.append(boxes_in_frame.pop(i))
                self.imgs.append(imgs_in_frame.pop(i))
                self.dates.append(date)
                self.track_len += 1

                # Update lower track border `lower_y`
                y = max(self.boxes[-1][1], self.boxes[-1][3])
                if y > self.lower_y:
                    self.lower_y = y
                break

        return masks_in_frame, boxes_in_frame, imgs_in_frame

    def _write_images(self):
        dir_name = str(id(self))
        if not os.path.isdir(dir_name):
            os.mkdir(dir_name)
        for idx, img in enumerate(self.imgs):
            cv2.imwrite(os.path.join(dir_name, f'{idx}.jpg'), img)


if __name__ == '__main__':
    pred = get_default_predictor()
    cd = CrossDetector(CachedPersonDetector('data/in_out/out_video.mp4', pred), 500)
    tbs = []
    for _ in range(150):
        tbs += cd.process_next_frame()
    for t in tbs:
        t._write_images()
    pass
