import numpy as np
import tensorflow as tf
from meta import ROOT_PATH
import os


def build(batch_size=0):
    d_path = os.path.join(ROOT_PATH, 'data', 'toy.npz')
    np_data = np.load(d_path)
    set_size = np_data['label'].__len__()

    dataset = tf.data.Dataset.from_tensor_slices({'data': np_data['data'], 'label': np_data['label']})

    if batch_size <= 0:
        batch_size = set_size

    rslt = dataset.shuffle(set_size).batch(batch_size, drop_remainder=True)

    return rslt


if __name__ == '__main__':
    d = build()
    # d = d.batch(4)

    i = iter(d)

    a = next(i)
    print(a)
