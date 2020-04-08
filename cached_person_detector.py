import numpy as np
import errno
import os

from person_detector import PersonDetector


class CachedPersonDetector(PersonDetector):

    def __init__(self, video_path, cache_dir='./'):
        super().__init__(video_path)
        if os.path.isdir(cache_dir):
            self.cache_dir = cache_dir
            self.cache_file_name = os.path.join(self.cache_dir, f'{self.video.video_name}__CACHE__.npz')
            if os.path.isfile(self.cache_file_name):
                cache = np.load(self.cache_file_name)
                self.detections_cache = cache['detections']
                self.masks_cache = cache['masks']
                self.masks_shapes = cache['shapes']
                self.cache_len = self.detections_cache[-1, 0].astype(int) + 1  # the last frame idx + 1
                # Reset counters
                self.cur_det_idx = 0
                self.cur_msk_idx = 0
                print(f'Length of loaded cache is {self.cache_len}')
            else:
                self._reset_cache()
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), cache_dir)

    def _reset_cache(self):
        self.detections_cache = np.array([[]])
        self.masks_cache = np.array([])
        self.masks_shapes = np.array([[]])
        self.cache_len = 0

    def process_next_frame(self):
        next_frame_num = self.video.frame_num + 1  # first frame has 0 idx
        if next_frame_num >= self.cache_len:  # frame not cached yet, inference needed
            date, frame, detections = super().process_next_frame()
            if detections is not None:
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
            return date, frame, detections
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
                if not any_detection:
                    return date, frame, None  # no detection in the current frame

                # Get as many masks as detections
                masks = []
                for shape in shapes:
                    size = shape[0] * shape[1]
                    masks.append(
                        np.array(self.masks_cache[self.cur_msk_idx:(self.cur_msk_idx + size)]).reshape(shape)
                    )
                    self.cur_msk_idx += size
                person_images = self.extract_person_images(frame, boxes)
                return date, frame, (masks, boxes, person_images, scores)
            else:  # end of video
                self.save_cache()
                raise EOFError

    def save_cache(self):
        np.savez(
            self.cache_file_name,
            detections=self.detections_cache,
            masks=self.masks_cache,
            shapes=self.masks_shapes
        )
        print(f"Cached saved to {self.cache_file_name}")


if __name__ == '__main__':
    import cv2
    from progiter import ProgIter
    pd = CachedPersonDetector('data/in_out/in_video.mp4')
    for i in ProgIter(range(150), verbose=3):
        date, f, detection_data = pd.process_next_frame()
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
    pd.save_cache()
