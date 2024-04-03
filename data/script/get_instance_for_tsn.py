import os
import json
import numpy as np
import pickle
from concurrent import futures

dataset = "../tiny-Kinetics-400"
frames_dir = dataset + "/frames"
label_files = {'train': 'label/train_256.json', 'val': 'label/val_256.json'}
with open(os.path.join(dataset, 'label/label.json'), 'r') as f:
    label_transform = json.load(f)


def process(item, save_folder, fps=30):
    action_pos = []
    url = ["{}/{}_{:0>6}_{:0>6}.mp4".format(a['label'],
                                            a['youtube_id'], a['time_start'], a['time_end']) for a in item]
    basename = [os.path.basename(a).split('.')[0] for a in url]

    actions = [{'label_ids': label_transform[sample['label']], 'label_names': sample['label'],
                'start_id': sample['time_start'], 'end_id': sample['time_end'], 'split': sample['split'],
                'video_id': sample['youtube_id'],
                'basename': os.path.basename("{}/{}_{:0>6}_{:0>6}.mp4".format(sample['label'],
                                                                              sample['youtube_id'],
                                                                              sample['time_start'],
                                                                              sample['time_end'])).split('.')[0]} for
               sample in item]
    for action in actions:
        action_pos.append(
            {
                'label': action['label_ids'],
                'label_name': action['label_names'],
                'video_name': action['video_id'],
                'start': 0,
                'end': (int(action['end_id']) - int(action['start_id'])) * fps,
                'basename': action['basename']
            })
    for item in action_pos:
        start = item['start']
        end = item['end']
        label = item['label']
        frames = []
        try:
            for ii in range(start + 1, end + 1):
                img = os.path.join(frames_dir, '{}_256_frames'.format(actions[0]['split']), item['label_name'],
                                                item['basename'],
                                'frame_{:0>5d}.jpg'.format(ii))
                with open(img, 'rb') as g:
                    data = g.read()
                frames.append(data)
            outname = '%s/%s_%s.pkl' % (save_folder, item['basename'], label)
            with open(outname, 'wb') as g:
                pickle.dump((item['basename'], label, frames), g, -1)
        except:
            continue


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
        # for i in label_data.values():
        #     process(i, save_folder)
    data_dir = '../tiny-Kinetics-400/input_for_tsn'
    a = os.listdir(data_dir + '/train')
    b = os.listdir(data_dir + '/val')
    with open(data_dir + '/train.list', 'w') as f:
        for item in a:
            f.write(os.path.join('train', item) + '\t' + item.split('_')[-1].split('.')[0] + '\n')
    with open(data_dir + '/val.list', 'w') as f:
        for item in b:
            f.write(os.path.join('val', item) + '\t' + item.split('_')[-1].split('.')[0] + '\n')
    # os.system('find ' + data_dir + 'train -name "*.pkl" > ' + data_dir +
    #           'train.list')
    # os.system('find ' + data_dir + 'val -name "*.pkl" > ' + data_dir +
    #           'val.list')


if __name__ == '__main__':
    run()
    # python -B -m paddle.distributed.launch --gpus="0" --log_dir=$save_dir/logs main.py --validate -c ../configs_train/pptsm.yaml -o output_dir=$save_dir
    # python tools/export_model.py -c ../configs)train/pptsm.yaml -p ../checkpoints/models_pptsm/ppTSM_epoch_00057.pdparams -o ../checkpoints/ppTSM
