from sys import getsizeof
import errno
import os

import numpy as np

from person_detector import PersonDetector, get_default_predictor


CACHE_PART_SIZE = 500 * 2**20  # max size of cache to hold in RAM


class CachedPersonDetector(PersonDetector):

    def __init__(self, video_path, predictor, cache_dir='./', decimation=None):
        super().__init__(video_path, predictor, decimation)
        if not os.path.isdir(cache_dir):
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cache_dir)
        self.cache_dir = cache_dir
        self.part_idx = -1
        self.no_more_detections_till_end = False
        self._load_next_part()

    def process_next_frame(self):
        next_frame_num = self.video.frame_num + 1  # first frame has 0 idx

        if next_frame_num >= self.cache_len \
                and not self.no_more_detections_till_end:  # frame not cached yet, inference needed

            retval = super().process_next_frame()

            if retval is not None:

                date, frame, detections = retval

                if detections is not None:  # append cache if any detection
                    # Create a new part if the cache's too big
                    if self.cache_size() > CACHE_PART_SIZE:
                        self._save_cache()
                        self._load_next_part()

                    masks, boxes, imgs, scores = detections
                    num_detections = scores.size

                    # Add new boxes and scores to the cache
                    new_cache = np.hstack((
                        np.full((num_detections, 1), self.video.frame_num), boxes, scores
                    ))
                    self.detections_cache = np.vstack((self.detections_cache, new_cache)) \
                        if self.detections_cache.size else new_cache

                    # Add new masks to the cache
                    for mask in masks:
                        self.masks_cache = np.append(self.masks_cache, mask.flatten())
                        shape = np.array(mask.shape).reshape(1, 2)
                        self.masks_shapes = np.append(self.masks_shapes, shape, axis=0) \
                            if self.masks_shapes.size else shape

                    self.cache_len += 1

                return date, frame, detections  # frame with or without detections

            else:
                self._save_cache()
                self._close_cache()
                return None  # end of video

        else:  # frame cached, return pre-calculated data

            retval = self.video.get_datetime_frame()

            if retval is not None:
                date, frame = retval

                # Get boxes, masks, and shapes
                boxes, scores, shapes = [], [], []
                any_detection = False
                while self.cur_det_idx < self.detections_cache.shape[0] \
                        and self.detections_cache[self.cur_det_idx, 0] == self.video.frame_num:
                    boxes.append(self.detections_cache[self.cur_det_idx, 1:5])
                    scores.append(self.detections_cache[self.cur_det_idx, 5])
                    shapes.append(self.masks_shapes[self.cur_det_idx])
                    self.cur_det_idx += 1
                    any_detection = True

                if any_detection:
                    # Get as many masks as detections
                    masks = []
                    for shape in shapes:
                        size = shape[0] * shape[1]
                        masks.append(
                            np.array(self.masks_cache[self.cur_msk_idx:(self.cur_msk_idx + size)]).reshape(shape)
                        )
                        self.cur_msk_idx += size
                    person_images = self.extract_person_images(frame, boxes)

                    if self.cur_det_idx >= self.detections_cache.shape[0]:
                        self._load_next_part()

                    return date, frame, (masks, boxes, person_images, scores)  # frame with detections
                else:
                    return date, frame, None  # frame without detections

            return None

    def cache_size(self):
        """In bytes"""
        return getsizeof(self.detections_cache) + getsizeof(self.masks_cache) + getsizeof(self.masks_shapes)

    def get_part_path(self):
        return os.path.join(self.cache_dir, f'{self.video.video_name}__CACHE_{self.part_idx:03d}_.npz')

    def _close_cache(self):
        # Write an empty cache part to close cache
        self.part_idx += 1
        self._reset_cache()
        self._save_cache()

    def _load_next_part(self):
        self.part_idx += 1
        part_path = self.get_part_path()
        if os.path.isfile(part_path):  # cache exists
            # load next part
            cache = np.load(part_path)
            self.detections_cache = cache['detections']
            self.masks_cache = cache['masks']
            self.masks_shapes = cache['shapes']

            if self.detections_cache.size == 0:  # reached the end of the cache
                self.no_more_detections_till_end = True
                return

            # the last frame idx + 1
            self.cache_len = self.detections_cache[-1, 0].astype(int) + 1

            # Reset counters
            self.cur_det_idx = 0
            self.cur_msk_idx = 0
            print(f'Loaded CACHE#{self.part_idx:03d}')
        else:
            self._reset_cache()

    def _reset_cache(self):
        self.detections_cache = np.array([[]])
        self.masks_cache = np.array([])
        self.masks_shapes = np.array([[]])
        self.cache_len = 0

    def _save_cache(self):
        np.savez_compressed(
            self.get_part_path(),
            detections=self.detections_cache,
            masks=self.masks_cache,
            shapes=self.masks_shapes
        )
        print(f'Saved  CACHE#{self.part_idx:03d}: {self.get_part_path()}')


if __name__ == '__main__':
    import cv2
    pd = CachedPersonDetector('data/test_video.mp4', get_default_predictor(), cache_dir='data')
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
