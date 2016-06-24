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
import model


def detect(im, param_vals):
    """
    Detect number plates in an image.

    :param im:
        Image to detect number plates in.

    :param param_vals:
        Model parameters to use. These are the parameters output by the `train`
        module.

    :returns:
        a 7,36 matrix giving the probability distributions of each letter.

    """

    # Load the model which detects number plates over a sliding window.
    x, y, params = model.get_detect_model()

    # Execute the model at each scale.
    with tf.Session(config=tf.ConfigProto()) as sess:
        feed_dict = {x: numpy.stack([im])}
        feed_dict.update(dict(zip(params, param_vals)))
        y_val = sess.run(y, feed_dict=feed_dict)
        hanzi_probs = (y_val[0, 0, 0, 1:32].reshape(1, len(common.HANZI)))
        hanzi_probs = common.softmax(hanzi_probs)
        
        letter_probs = (y_val[0, 0, 0, 32:58].reshape(1, len(common.LETTERS)))
        letter_probs = common.softmax(letter_probs)
        
        char_probs = (y_val[0, 0, 0, 58:].reshape(5, len(common.CHARS)))
        char_probs = common.softmax(char_probs)
        
        present_prob = common.sigmoid(y_val[0, 0, 0, 0])
        
        return present_prob, char_probs, hanzi_probs, letter_probs



def char_probs_to_code(hanzi_probs, letter_probs, char_probs):
    return "".join([common.HANZI[numpy.argmax(hanzi_probs, axis=1)] , common.LETTERS[numpy.argmax(letter_probs, axis=1)], "".join(common.CHARS[i] for i in numpy.argmax(char_probs, axis=1)) ])


if __name__ == "__main__":
    #input1 = "./LPImages/UK1.jpg"
    #input1 = "./LPImages/car11.bmp"
    input1 = "./LPImages/source3.png"
    input2 = "./TrainedWeights/weights104831.npz"
    output1 = "./LPImages/carOut1.jpg"
    
    im = cv2.imread(input1)
    im_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) / 255.

    f = numpy.load(input2)
    param_vals = [f[n] for n in sorted(f.files, key=lambda s: int(s[4:]))]

    present_prob, char_probs, hanzi_probs, letter_probs = detect(im_gray, param_vals)
    code = char_probs_to_code(hanzi_probs, letter_probs, char_probs)
    print present_prob
    print code


