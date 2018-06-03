import tensorflow as tf
import numpy as np
import time

batch_size = 1024
filename_queue = tf.train.string_input_producer(["train.csv"])

reader = tf.TextLineReader(skip_header_lines=1)
key, value = reader.read(filename_queue)


record_defaults = [tf.constant([0], dtype=tf.int32),    # Column 1
                   tf.constant([0], dtype=tf.int32),    # Column 2
                   tf.constant([0], dtype=tf.int32),    # Column 3
                   tf.constant([0], dtype=tf.int32),    # Column 4

                    tf.constant([0], dtype=tf.int32),    # Column 5
                    tf.constant([' '], dtype=tf.string),   # Column 6
                    tf.constant([' '], dtype=tf.string),    # Column 7
                    tf.constant([0], dtype=tf.int32)]    # Column 8
col1, col2, col3, col4, col5, col6, col7, col8 = tf.decode_csv(
    value, record_defaults=record_defaults)

days = tf.string_split([col6], delimiter=' ').values[0]
days = tf.string_split([days], delimiter='-').values[2]
days = tf.string_to_number(days)
days = tf.cast(days, tf.int32)

dates = tf.string_split([col6], delimiter=' ').values[1]

hours = tf.string_split([dates], delimiter=':').values[0]
hours = tf.string_to_number(hours)
hours = tf.cast(hours, tf.int32)

minutes = tf.string_split([dates], delimiter=':').values[1]
minutes = tf.string_to_number(minutes)
minutes = tf.cast(minutes, tf.int32)

features = tf.stack([col1, col2, col3, col4, col5, days, hours, minutes])

# days = tf.string_split([days], delimiter='-').values[2]
# hours = tf.string_split([hours], delimiter='-').values[0]

# dates = tf.stack([days, hours])
# dates = tf.stack(col6)

print(col6)

min_after_dequeue = 1000
capacity = min_after_dequeue + 3 * batch_size

example_batch, label_batch = tf.train.shuffle_batch(
  [features, col8], num_threads=32, batch_size=batch_size, capacity=capacity,
  min_after_dequeue=min_after_dequeue)

with tf.Session() as sess:

  # Start populating the filename queue.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  print(time.clock())

  for i in range(10000):
    # Retrieve a single instance:


    example, label = sess.run([example_batch, label_batch])

    if i % 100 == 0:
        print(time.time())
        print(example.shape, label.shape)
        print(example[0,:], label[0])



  coord.request_stop()
  coord.join(threads)


