import numpy as np


from cached_person_detector import CachedPersonDetector


class CrossDetector:

    def __init__(self, video_path, gate_y_coor, gate_width=100):  # 650
        self.person_detector = CachedPersonDetector(video_path)
        self.gate_y_coor = gate_y_coor
        self.gate_width = gate_width
        self.track_boxes = []

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
                if tb.lower_y > self.gate_y_coor + self.gate_width:
                    out.append(tb)  # return a complete `TrackingBox`
                    self.track_boxes.remove(tb)
            for mask, box, img in zip(masks, boxes, images):
                _, y1, _, y2 = box
                lower_y = max(y1, y2)
                if self.gate_y_coor < lower_y < self.gate_y_coor + self.gate_width:
                    self.track_boxes.append(TrackingBox(mask, box, img))
        return out


class TrackingBox:

    IOU_THRESH = 0.9

    def __init__(self, init_mask, init_box, init_img, init_date):
        self.masks = [init_mask]
        self.boxes = [init_box]
        self.imgs = [init_img]
        self.dates = [init_date]
        self.track_len = 1
        self.lower_y = max(init_box[1], init_box[3])  # the lower point has maximum y coordinate

    def IoU_with_mask(self, mask, box):
        if self.is_intersection(box):
            sbox = self.boxes[-1]  # self box
            min_x = min(box[0], sbox[0])
            min_y = min(box[1], sbox[1])
            smask_with_padding = np.full((h_px, w_px), False)
            mask_with_padding = np.full((h_px, w_px), False)
            # sbox = self.boxes[-1]  # self box
            # smask = self.mask[-1].copy()  # self mask
            # mask = mask.copy()
            #
            # # Expansion size
            # w_px = np.round(max(box[2], sbox[2]) - min(box[0], sbox[0])).astype("int")
            # h_px = np.round(max(box[3], sbox[3]) - min(box[1], sbox[1])).astype("int")
            #
            # smask_with_padding = np.full((h_px, w_px), False)
            # mask_with_padding = np.full((h_px, w_px), False)
            # if max(box[1], box[3]) > max(sbox[1], sbox[3]):
            #     if max(box[0], box[2]) > max(sbox[0], sbox[2]):
            #         #   ----------
            #         #  |smask     |
            #         #  |          |
            #         #  |      mask|
            #         #   ----------
            #         mask_with_padding[-mask.shape[0]:, -mask.shape[1]:] =
            #         smask_with_padding[] =
            #     else:  # sbox to the right of the box
            #         #   ----------
            #         #  |          |
            #         #  |          |
            #         #  |          |
            #         #   ----------
            #         pass
            # else:  # sbox lower than box
            #     if max(box[0], box[2]) > max(sbox[0], sbox[2]):  # box to the right of the sbox
            #         #   ----------
            #         #  |          |
            #         #  |          |
            #         #  |          |
            #         #   ----------
            #         pass
            #     else:  # sbox to the right of the box
            #         #   ----------
            #         #  |          |
            #         #  |          |
            #         #  |          |
            #         #   ----------
            #         pass

            return (self.masks[-1] & mask).sum() / (self.masks[-1] | mask).sum()
        else:
            return 0

    def is_intersection(self, box):
        x1, y1, x2, y2 = box
        x3, y3, x4, y4 = self.boxes[-1]
        assert x2 > x1 and x4 > x3
        x_cross = TrackingBox.intersection_1d(x1, x2, x3, x4)
        assert y2 > y1 and y4 > y3
        y_cross = TrackingBox.intersection_1d(y1, y2, y3, y4)
        return x_cross * y_cross > 0

    @staticmethod
    def intersection_1d(start1, end1, start2, end2):
        start = max(start1, start2)
        end = min(end1, end2)
        if start < end:
            return end - start
        else:
            return 0

    def update(self, masks_in_frame, boxes_in_frame, imgs_in_frame, date):
        for mask, box, img in zip(masks_in_frame, boxes_in_frame, imgs_in_frame):
            if self.IoU_with_mask(mask, box) > self.IOU_THRESH:
                # Append a new detection to the track
                self.masks.append(mask)
                self.boxes.append(box)
                self.imgs.append(img)
                self.dates.append(date)
                self.track_len += 1

                # Update lower track border `lower_y`
                y = max(box[1], box[3])
                if y > self.lower_y:
                    self.lower_y = y

                # Remove used detection from list of all ones
                masks_in_frame.remove(mask)
                boxes_in_frame.remove(box)
                imgs_in_frame.remove(img)
                break
        return masks_in_frame, boxes_in_frame, imgs_in_frame


if __name__ == '__main__':
    cd = CrossDetector('data/in_out/in_video.mp4', 650)
    for _ in range(150):
        cd.process_next_frame()
    pass
