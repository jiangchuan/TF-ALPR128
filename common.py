#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
Definitions that don't fit elsewhere.

"""

__all__ = (
    'DIGITS',
    'LETTERS',
    'HANZI',
    'DOT',
    'CHARS',
    'ALLCHARS',
    'sigmoid',
    'softmax',
)

import numpy


DIGITS = unicode("0123456789", 'utf8')
LETTERS = unicode("ABCDEFGHIJKLMNOPQRSTUVWXYZ", 'utf8')
HANZI = unicode("京津沪渝冀豫云辽黑湘皖鲁苏赣浙粤鄂桂甘晋蒙陕吉闽贵青藏川宁新琼", 'utf8')
DOT = u"\u00B7"

CHARS = LETTERS + DIGITS
ALLCHARS = HANZI + CHARS

def softmax(a):
    exps = numpy.exp(a.astype(numpy.float64))
    return exps / numpy.sum(exps, axis=-1)[:, numpy.newaxis]

def sigmoid(a):
  return 1. / (1. + numpy.exp(-a))

