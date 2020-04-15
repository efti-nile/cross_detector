import cv2
import os
import pytz
import time
from datetime import datetime, timedelta
import subprocess


class VideoReader:
    """Reads video with a datetime for each frame.

    If there is {video_name}.datetime text file, the start datetime is read from it.
    Example of datetime string: '2020-04-01T12:00:00 UTC'

    Otherwise the start datetime is read by hachoir-metadata utility.
    """

    def __init__(self, filepath, decimation=None):
        self.cap = cv2.VideoCapture(filepath)
        dirpath, filename = os.path.split(filepath)
        self.video_name = filename.split('.')[0]
        self.frame_num = -1
        datetime_path = os.path.join(dirpath, f'{self.video_name}.datetime')
        if decimation is not None:
            assert isinstance(decimation, int) and decimation >= 1
            self.frames_to_grab = decimation - 1
        else:
            self.frames_to_grab = 0
        if os.path.exists(datetime_path):
            with open(datetime_path) as f:
                datetime_str = f.readline().strip()
                self.start_datetime = datetime.strptime(datetime_str, '%Y-%m-%dT%H:%M:%S %Z')
        else:
            self.start_datetime = VideoReader.get_start_datetime_meta(filepath)

    @staticmethod
    def get_start_datetime_meta(filename):
        result = subprocess.Popen(['hachoir-metadata', filename],
                                  stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        results = result.stdout.read().decode('utf-8').split('\n')
        start_datetime = None

        for item in results:
            if item.startswith('- Creation date: '):
                start_datetime_str = item.lstrip('- Creation date: ')
                # start_datetime = datetime.datetime.strptime(start_datetime_str, '%Y-%m-%d %H:%M:%S')
                start_datetime = datetime(*(time.strptime(start_datetime_str, '%Y-%m-%d %H:%M:%S')[0:6]),
                                          tzinfo=pytz.timezone("US/Pacific"))  # with timezone
                break

        return start_datetime

    def get_datetime_frame(self):
        for _ in range(self.frames_to_grab):
            self.cap.grab()
        frame_returned, frame = self.cap.read()
        time_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        dt = self.start_datetime + timedelta(milliseconds=time_ms)
        if frame_returned:
            self.frame_num += 1
            return dt, frame
        else:
            return None

    def __del__(self):
        self.cap.release()


if __name__ == '__main__':
    vo = VideoReader('data/in_out/in_video.mp4')
    tmp = vo.get_datetime_frame()
    i = 0
    while tmp is not None:
        print(i)
        i += 1
        print(tmp[0])
        tmp = vo.get_datetime_frame()
    pass

