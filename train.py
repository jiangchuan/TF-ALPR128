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
Routines for training the network.

"""


__all__ = (
    'train',
)


import functools
import glob
import itertools
import multiprocessing
import random
import sys
import time

import cv2
import numpy
import tensorflow as tf

from tensorflow.python.client import graph_util
from tensorflow.python.platform import gfile


import common
import gen
import model


def code_to_vec(p, code):
    def char_to_vec(c, theStr):
        y = numpy.zeros((len(theStr),))
        y[theStr.index(c)] = 1.0
        return y
    
    chars = numpy.vstack([char_to_vec(c, common.CHARS) for c in code[2:]])
    return numpy.concatenate([[1. if p else 0], char_to_vec(code[0], common.HANZI), char_to_vec(code[1], common.LETTERS), chars.flatten()])


def read_data(img_glob):
    for fname in sorted(glob.glob(img_glob)):
        im = cv2.imread(fname.encode('utf8', 'replace'))[:, :, 0].astype(numpy.float32) / 255.
        #cv2.imshow('image',im)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()        
        code = fname.split("/")[1][9:16]
        p = fname.split("/")[1][17] == '1'
        yield im, code_to_vec(p, code)


def unzip(b):
    xs, ys = zip(*b)
    xs = numpy.array(xs)
    ys = numpy.array(ys)
    return xs, ys


def batch(it, batch_size):
    out = []
    for x in it:
        out.append(x)
        if len(out) == batch_size:
            yield out
            out = []
    if out:
        yield out


def mpgen(f):
    def main(q, args, kwargs):
        try:
            for item in f(*args, **kwargs):
                q.put(item)
        finally:
            q.close()

    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        q = multiprocessing.Queue(3) 
        proc = multiprocessing.Process(target=main,
                                       args=(q, args, kwargs))
        proc.start()
        try:
            while True:
                item = q.get()
                yield item
        finally:
            proc.terminate()
            proc.join()

    return wrapped
        

@mpgen
def read_batches(batch_size):
    def gen_vecs():
        for im, c, p in gen.generate_ims(batch_size):
            yield im, code_to_vec(p, c)

    while True:
        yield unzip(gen_vecs())


def train(learn_rate, report_steps, save_steps, batch_size, initial_weights=None):
    """
    Train the network.

    The function operates interactively: Progress is reported on stdout, and
    training ceases upon `KeyboardInterrupt` at which point the learned weights
    are saved to `weights.npz`, and also returned.

    :param learn_rate:
        Learning rate to use.

    :param report_steps:
        Every `report_steps` batches a progress report is printed.

    :param batch_size:
        The size of the batches used for training.

    :param initial_weights:
        (Optional.) Weights to initialize the network with.

    :return:
        The learned network weights.

    """
    x, y, params = model.get_training_model()

    numNeuron = 1 + len(common.HANZI) + len(common.LETTERS) + 5 * len(common.CHARS)
    y_ = tf.placeholder(tf.float32, [None, numNeuron])

    char_loss = tf.nn.softmax_cross_entropy_with_logits(
                                          tf.reshape(y[:, 58:],
                                                     [-1, len(common.CHARS)]),
                                          tf.reshape(y_[:, 58:],
                                                     [-1, len(common.CHARS)]))
    char_loss = tf.reduce_sum(char_loss)
  
    hanzi_loss = tf.nn.softmax_cross_entropy_with_logits(
                                          tf.reshape(y[:, 1:32],
                                                     [-1, len(common.HANZI)]),
                                          tf.reshape(y_[:, 1:32],
                                                     [-1, len(common.HANZI)]))
    hanzi_loss = tf.reduce_sum(hanzi_loss)

    letter_loss = tf.nn.softmax_cross_entropy_with_logits(
                                          tf.reshape(y[:, 32:58],
                                                     [-1, len(common.LETTERS)]),
                                          tf.reshape(y_[:, 32:58],
                                                     [-1, len(common.LETTERS)]))
    letter_loss = tf.reduce_sum(letter_loss)
  
    #presence_loss = 10. * tf.nn.sigmoid_cross_entropy_with_logits(y[:, :1], y_[:, :1])
    presence_loss = tf.nn.sigmoid_cross_entropy_with_logits(y[:, :1], y_[:, :1])
    presence_loss = tf.reduce_sum(presence_loss)
    cross_entropy = char_loss + hanzi_loss + letter_loss + presence_loss
    train_step = tf.train.AdamOptimizer(learn_rate).minimize(cross_entropy)

    char_best = tf.argmax(tf.reshape(y[:, 58:], [-1, 5, len(common.CHARS)]), 2)
    char_correct = tf.argmax(tf.reshape(y_[:, 58:], [-1, 5, len(common.CHARS)]), 2)

    hanzi_best = tf.argmax(tf.reshape(y[:, 1:32], [-1, 1, len(common.HANZI)]), 2)
    hanzi_correct = tf.argmax(tf.reshape(y_[:, 1:32], [-1, 1, len(common.HANZI)]), 2)

    letter_best = tf.argmax(tf.reshape(y[:, 32:58], [-1, 1, len(common.LETTERS)]), 2)
    letter_correct = tf.argmax(tf.reshape(y_[:, 32:58], [-1, 1, len(common.LETTERS)]), 2)

    if initial_weights is not None:
        assert len(params) == len(initial_weights)
        assign_ops = [w.assign(v) for w, v in zip(params, initial_weights)]

    init = tf.initialize_all_variables()

    def vec_to_plate(v):
        return "".join(common.CHARS[i] for i in v)

    def do_report():
        r = sess.run([char_best,
                      char_correct,
                      tf.greater(y[:, 0], 0),
                      y_[:, 0],
                      char_loss,
                      presence_loss,
                      cross_entropy,
                      hanzi_best,
                      hanzi_correct,
                      letter_best,
                      letter_correct,
                      hanzi_loss,
                      letter_loss],
                     feed_dict={x: test_xs, y_: test_ys})
                     
        plate_correct = numpy.logical_and(numpy.all(r[0] == r[1], axis=1),numpy.logical_and(numpy.all(r[7] == r[8], axis=1),numpy.all(r[9] == r[10], axis=1)))
        #num_correct = numpy.sum(numpy.logical_or(plate_correct, numpy.logical_and(r[2] < 0.5, r[3] < 0.5)))
        num_correct = numpy.sum(plate_correct)
        num_p_correct = numpy.sum(r[2] == r[3])

        print ("Batch {:3d}: {:2.02f}%, {:2.02f}%, loss: {:.1f} "
               "(hanzi: {:.1f}, letters: {:.1f}, chars: {:.1f}, presence: {:.1f})").format(
            batch_idx,
            100. * num_correct / (len(r[0])),
            100. * num_p_correct / len(r[2]),
            r[6],
            r[11],
            r[12],
            r[4],
            r[5])

    def save_graph():
        # Write out the trained graph and labels with the weights stored as constants.
        output_graph_def = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), ['final_result'])
        with gfile.FastGFile('./Graph/output_graph.pb', 'wb') as f:
            f.write(output_graph_def.SerializeToString())


    def save_weights():
        print ("Saving weights from batch {:3d} ...").format(batch_idx)
        last_weights = [p.eval() for p in params]
        numpy.savez("./TrainedWeights/weights" + `batch_idx` + ".npz", *last_weights)
        
        #tf.train.write_graph(sess.graph_def, "/tmp/load", "test.pb", False) #proto

        return last_weights


    def do_batch():
        sess.run(train_step,
                 feed_dict={x: batch_xs, y_: batch_ys})
        if batch_idx % report_steps == 0:
            do_report()
        if (batch_idx % save_steps == 0 and batch_idx > 0):
            save_graph()
            save_weights()



    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        if initial_weights is not None:
            sess.run(assign_ops)

        test_xs, test_ys = unzip(list(read_data(u"test/*.png"))[:50])

        try:
            last_batch_idx = 0
            last_batch_time = time.time()
            batch_iter = enumerate(read_batches(batch_size))
            for batch_idx, (batch_xs, batch_ys) in batch_iter:
                do_batch()
                if batch_idx % report_steps == 0:
                    batch_time = time.time()
                    if last_batch_idx != batch_idx:
                        print "time for 60 batches {}".format(
                            60 * (last_batch_time - batch_time) /
                                            (last_batch_idx - batch_idx))
                        last_batch_idx = batch_idx
                        last_batch_time = batch_time

        except KeyboardInterrupt:
            save_graph()
            return save_weights()


if __name__ == "__main__":
    load_initial_weights = True
    input2 = "./TrainedWeights/weights104831.npz"

    if load_initial_weights:
        f = numpy.load(input2)
        initial_weights = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]
    else:
        initial_weights = None

    last_weights = train(learn_rate=0.001/2,
          report_steps=60,
          save_steps=1,
          batch_size=50,
          initial_weights=initial_weights)

