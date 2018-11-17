# -*- coding: utf-8 -*-
import pickle

def get_num_lines(filename):
    count = 0
    with open(filename, 'rb') as file: 
        while 1:
            buffer = file.read(8192*1024)
            if not buffer: break
            count += buffer.count(b'\n')
    return count

def hms_string(sec_elapsed):
    h = int(sec_elapsed / (60 * 60))
    m = int((sec_elapsed % (60 * 60)) / 60)
    s = sec_elapsed % 60
    return "{}:{:>02}:{:>05.2f}".format(h, m, s)

def save_vocabulary(filename, dict_object):
    with open(filename, 'wb') as f:
        pickle.dump(dict_object, f, pickle.HIGHEST_PROTOCOL)

def load_vocabulary(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)