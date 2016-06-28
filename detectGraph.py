#!/usr/bin/env python
#
# Copyright (c) 2016 Matthew Earl
# 
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
# 
#     The above copyright notice and this permission notice shall be included
#     in all copies or substantial portions of the Software.
# 
#     THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
#     OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#     MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN
#     NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
#     DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
#     OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE
#     USE OR OTHER DEALINGS IN THE SOFTWARE.


"""
Routines to detect number plates.

Use `detect` to detect all bounding boxes, and use `post_process` on the output
of `detect` to filter using non-maximum suppression.

"""


__all__ = (
    'detect',
)


import cv2
import numpy
import tensorflow as tf

import common


def create_graph(graph_path):
  """Creates a graph from saved GraphDef file and returns a saver."""
  # Creates graph from saved graph_def.pb.
  with tf.gfile.FastGFile(graph_path, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    _ = tf.import_graph_def(graph_def, name='')


def detect(image, graph_path):
  """Runs inference on an image.

  Args:
    image: Image file name.

  Returns:
    Nothing
  """
  if not tf.gfile.Exists(image):
      tf.logging.fatal('File does not exist %s', image)
  image_data = tf.gfile.FastGFile(image, 'rb').read()

  # Creates graph from saved GraphDef.
  create_graph(graph_path)

  with tf.Session() as sess:
    # Some useful tensors:
    # 'softmax:0': A tensor containing the normalized prediction across
    #   1000 labels.
    # 'pool_3:0': A tensor containing the next-to-last layer containing 2048
    #   float description of the image.
    # 'DecodeJpeg/contents:0': A tensor containing a string providing JPEG
    #   encoding of the image.
    # Runs the softmax tensor by feeding the image_data as input to the graph.
  
    #x = sess.graph.get_tensor_by_name('GreyImageInput:0')
    present_prob = sess.graph.get_tensor_by_name('present_prob:0')
    hanzi_prob = sess.graph.get_tensor_by_name('hanzi_prob:0')
    letter_prob = sess.graph.get_tensor_by_name('letter_prob:0')
    chars_prob = sess.graph.get_tensor_by_name('chars_prob:0')

    val_present_prob, val_hanzi_prob, val_letter_prob, val_chars_prob = sess.run([present_prob, hanzi_prob, letter_prob, chars_prob], {'GreyImageInput:0': image_data})
    #val_present_prob, val_hanzi_prob, val_letter_prob, val_chars_prob = sess.run([present_prob, hanzi_prob, letter_prob, chars_prob], {x: numpy.stack([im])})
    return val_present_prob, val_hanzi_prob, val_letter_prob, val_chars_prob



def char_probs_to_code(hanzi_probs, letter_probs, char_probs):
    return "".join([common.HANZI[numpy.argmax(hanzi_probs, axis=1)] , common.LETTERS[numpy.argmax(letter_probs, axis=1)], "".join(common.CHARS[i] for i in numpy.argmax(char_probs, axis=1)) ])



if __name__ == "__main__":
    graph_path = './Graph/ALPR_graph.pb'
    image_path = "./LPImages/source1.png"
    
    #im = cv2.imread(image_path)
    #im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.

    present_prob, char_probs, hanzi_probs, letter_probs = detect(image_path, graph_path)
    code = char_probs_to_code(hanzi_probs, letter_probs, char_probs)
    print present_prob
    print code


