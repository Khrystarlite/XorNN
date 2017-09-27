import tensorflow as tf
import os



reader = tf.TextLineReader()

filename_queue = tf.train.string_input_producer(["Data/test.txt", "Data/labels/test_targets", "Data/train.txt", "Data/labels/test_targets.txt"])