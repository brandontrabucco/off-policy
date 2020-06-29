import tensorflow as tf


class Logger(object):

    def __init__(self,
                 logging_dir):
        """Creates a logging interface to a tensorboard file for
        visualizing in the tensorboard web interface; note that
        mean, max, min, and std are recorded

        Arguments:

        logging_dir: str
            the path on the disk to save records to"""

        tf.io.gfile.makedirs(logging_dir)
        self.writer = tf.summary.create_file_writer(logging_dir)

    def record(self,
               key,
               value,
               step):
        """Log statistics about training data to tensorboard
        log files for visualization later

        Arguments:

        key: str
            the string name to use when logging data in tensorboard
            that determines groupings in the web interface
        value: tf.tensor
            the tensor of values to record statistics about
            typically is multi dimensional
        step: int
            the total number of environment steps collected so far
            typically on intervals of 10000"""

        with self.writer.as_default():

            # log several statistics of the incoming tensors
            tf.summary.scalar(key + '/mean',
                              tf.math.reduce_mean(value),
                              step=step)
            tf.summary.scalar(key + '/max',
                              tf.math.reduce_max(value),
                              step=step)
            tf.summary.scalar(key + '/min',
                              tf.math.reduce_min(value),
                              step=step)
            tf.summary.scalar(key + '/std',
                              tf.math.reduce_std(value),
                              step=step)
