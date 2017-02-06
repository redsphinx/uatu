import tensorflow as tf

path = '/home/gabi/Documents/datasets/humans/1/per00001.jpg'

file_queue = tf.train.string_input_producer([path])
print(type(file_queue))

reader = tf.WholeFileReader()
print(type(reader))

key, value = reader.read(file_queue)
print(key)
print(value)

my_img = tf.image.decode_png(value)
print(type(my_img))