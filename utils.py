#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2022/10/9 22:06
# @Author  : hbl
# @File    : utils.py
# @Desc    :


from contextlib import contextmanager
from timeit import default_timer
from functools import wraps
import os





def elpased_time(a_func):
    @wraps(a_func)
    def wrap_func(*args, **kwargs):
        start = default_timer()
        res = a_func(*args, **kwargs)
        print('【'+a_func.__name__ + '】 elpased time：{}s'.format('%.6f' % (default_timer() - start)))
        return res
    return wrap_func

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

def read_paral_data(path, filename, langs):
    data = {}
    with open(os.path.join(path, f'{filename}.{langs[0]}'), encoding='utf-8') as f0:
        data[langs[0]] = [line.strip() for line in f0]

    with open(os.path.join(path, f'{filename}.{langs[1]}'), encoding='utf-8') as f1:
        data[langs[1]] = [line.strip() for line in f1]
    return data



# if __name__ == '__main__':
#     with elapsed_timer() as elapsed:
#         print('耗时：{}s'.format('%.6f' % elapsed()))