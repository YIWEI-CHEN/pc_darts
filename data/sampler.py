import glob

import numpy as np
import os
from shutil import copyfile


def copy(src, dst, files):
    if not os.path.exists(dst):
        os.makedirs(dst)
    for f in files:
        src_f = os.path.join(src, f)
        dst_f = os.path.join(dst, f)
        if not os.path.exists(dst_f):
            copyfile(src_f, dst_f)


if __name__ == '__main__':
    seed = 42
    np.random.seed(seed)

    root = '/data/yiwei/'
    src = 'imagenet'
    dst = 'imagenet_search'

    train,  train_portion = 'train', 0.1
    val, val_portion = 'val', 0.025
    total_portion = train_portion + val_portion

    train_path = os.path.join(root, src, train)
    for i, c in enumerate(os.listdir(train_path)):
        src_path = os.path.join(train_path, c)
        if not os.path.isdir(src_path):
            continue
        file_list = os.listdir(src_path)
        num_samples = int(round(len(file_list) * total_portion, 0))
        selected_files = np.random.choice(file_list, num_samples, replace=False)
        print('{:04}-{} selects {:03} images'.format(i, c, len(selected_files)))
        # selected_files.sort()
        split = int(round(len(file_list) * train_portion))

        # training samples
        train_list = selected_files[:split]
        dst_path = os.path.join(root, dst, train, c)
        copy(src_path, dst_path, train_list)

        # validation samples
        valid_list = selected_files[split:]
        dst_path = os.path.join(root, dst, val, c)
        copy(src_path, dst_path, valid_list)
