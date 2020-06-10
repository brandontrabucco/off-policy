import tensorflow as tf
import os
from skvideo.io import FFmpegWriter


class VideoSaver(object):

    def __init__(self,
                 save_folder,):
        """Create a training interface for an rl agent using
        the provided rl algorithm

        Args:

        save_folder: str
            a string that points to a specific folder on the device
        """

        self.save_folder = save_folder
        self.writer = None
        tf.io.gfile.makedirs(self.save_folder)

    def _open(self, number):
        self.writer = FFmpegWriter(
            os.path.join(self.save_folder, f"{number}.mp4"))

    @tf.function
    def open(self, number):
        tf.py_function(self._open, [number], [])

    def _write_frame(self, frame):
        self.writer.writeFrame(frame)

    @tf.function
    def write_frame(self, frame):
        tf.py_function(self._write_frame, [frame], [])

    def _close(self):
        self.writer.close()

    @tf.function
    def close(self):
        tf.py_function(self._close, [], [])
