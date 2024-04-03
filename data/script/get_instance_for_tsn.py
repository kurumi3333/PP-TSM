import os
import json
import subprocess

import numpy as np
import random
import pickle
from concurrent import futures
from tqdm import tqdm

dataset = "../tiny-Kinetics-400"
frames_dir = dataset + "/frames"
label_files = {'train': 'label/train_256.json', 'val': 'label/val_256.json'}


def process(item, save_folder, fps=30):
    action_pos = []
    url = ["{}/{}_{:0>6}_{:0>6}.mp4".format(a['label'],
                                            a['youtube_id'], a['time_start'], a['time_end']) for a in item]
    basename = [os.path.basename(a).split('.')[0] for a in url]

    action = [{'label': sample['label'], 'start': sample['start'] * fps, 'end': sample['end'] * fps} for sample in item]

    for item in tqdm(action):
        start = item['start']
        end = item['end']
        label = item['label']
        frames = []
        for ii in range(start + 1, end + 1):
            img = os.path.join(frames_dir, 'train_256_frames', basename, 'frame_{:06d}.jpg'.format(ii))
            with open(img, 'rb') as f:
                data = f.read()
            frames.append(data)
        outname = '%s/%s_%s.pkl' % (save_folder, basename, label)
        with open(outname, 'wb') as f:
            pickle.dump((basename, label, frames), f, -1)


def gen_instance_pkl(label_data, save_folder, fps=30):
    gts = label_data.values()
    with futures.ProcessPoolExecutor(max_workers=8) as executor:
        fs = [executor.submit(process, gt, save_folder, fps) for gt in gts]


def run():
    for item, value in label_files.items():
        save_folder = os.path.join(dataset, 'input_for_tsn', item)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        label_file = os.path.join(dataset, value)
        with open(label_file, 'r') as f:
            label_data = json.load(f)

        gen_instance_pkl(label_data, save_folder)
    data_dir = '/data/tiny-Kinetics-400/input_for_tsn'
    os.system('find ' + data_dir + 'train -name "*.pkl" > ' + data_dir + 'train.list')
    os.system('find ' + data_dir + 'val -name "*.pkl" > ' + data_dir + 'val.list')


if __name__ == '__main__':
    run()
