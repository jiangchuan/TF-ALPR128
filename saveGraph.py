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
    'save_graph',
)


import numpy

import tensorflow as tf
from tensorflow.python.client import graph_util
from tensorflow.python.platform import gfile

import common
import model


def save_graph(weight_path, graph_path):
    """
    Save the graph.

    The function operates interactively: Progress is reported on stdout, and
    training ceases upon `KeyboardInterrupt` at which point the learned weights
    are saved to `weights.npz`, and also returned.

    :param initial_weights:
        (Optional.) Weights to initialize the network with.
    """
    
    # Load the model which detects number plates over a sliding window.
    x, y, params = model.get_flat_detect_model()
    present_logits = tf.slice(y, [0, 0], [-1, 1])
    hanzi_logits = tf.slice(y, [0, 1], [-1, len(common.HANZI)])
    letter_logits = tf.slice(y, [0, 32], [-1, len(common.LETTERS)])
    chars_logits = tf.slice(y, [0, 58], [-1, 5 * len(common.CHARS)])

    present_prob = tf.sigmoid(present_logits, name='present_prob')
    hanzi_prob = tf.nn.softmax(hanzi_logits, name='hanzi_prob')
    letter_prob = tf.nn.softmax(letter_logits, name='letter_prob')
    chars_prob = tf.nn.softmax(tf.reshape(chars_logits, [5, len(common.CHARS)]), name='chars_prob')

    fw = numpy.load(weight_path)
    initial_weights = [fw[n] for n in sorted(fw.files, key=lambda s: int(s[4:]))]
    assert len(params) == len(initial_weights)
    assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)]

    init = tf.initialize_all_variables()

    #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    with tf.Session(config=tf.ConfigProto()) as sess:
        sess.run(init)
        sess.run(assign_ops)
        # Write out the trained graph and labels with the weights stored as constants.
        output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['present_prob', 'hanzi_prob', 'letter_prob', 'chars_prob'])
        with gfile.FastGFile(graph_path, 'wb') as f:
            f.write(output_graph_def.SerializeToString())



if __name__ == "__main__":
    weight_path = './TrainedWeights/weights126610.npz'
    graph_path = './Graph/ALPR_graph.pb'
    save_graph(weight_path, graph_path)


