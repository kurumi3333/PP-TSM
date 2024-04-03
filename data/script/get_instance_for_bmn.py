import os
import sys
import json
import random
import pickle
import numpy as np

bmn_window = 10
dataset = "data/tiny-Kinetics-400"
feat_dir = dataset + "/features"
out_dir = dataset + "/input_for_bmn"

label_files = {
    'train': 'label/train_256.json',
    'validation': 'label/val_256.json'
}

global fps


def gen_gts_for_bmn(gts_data):
    fps = 30
    gts_bmn = {'fps': fps, 'gts': []}
    for sub_item in gts_data:
        max_length = (int(sub_item['time_end']) - int(sub_item['time_start'])) * fps
        gts_bmn['gts'].append({
            'url': sub_item['youtube_id'],
            'total_frames': max_length,
            'root_actions': sub_item['label'],
            'time_start': sub_item['time_start'],
            'time_end': sub_item['time_end'],
            'label': sub_item['label']
        })
    gts_bmn = gts_bmn['gts']
    return gts_bmn


def save_feature_to_numpy(gts_bmn, folder):
    global fps
    print('save feature for bmn ...')
    if not os.path.exists(folder):
        os.mkdir(folder)
    process_gts_bmn = {}
    for item in gts_bmn:
        basename = item['url'] + '_' + item['time_start'] + '_' + item['time_end']
        time_start = item['time_start']
        time_end = item['time_end']
        if not basename in process_gts_bmn:
            process_gts_bmn[basename] = []
        process_gts_bmn[basename].append({
            'name': item['label'] + '/' + item['url'],
            'start': int(time_start),
            'end': int(time_end)
        })
    for item, values in process_gts_bmn.items():
        print(item, values)
        feat_path = os.path.join(feat_dir, item + '.pkl')
        print(feat_path)
        feature_video = pickle.load(open(feat_path, 'rb'))['image_feature']
        for value in values:
            save_cut_name = os.path.join(folder, value['name'])
            start_frame = ((value['start'] - 1) * fps)
            end_frame = ((value['end'] - 1) * fps)
            if end_frame > len(feature_video):
                continue
            feature_cut = [
                feature_video[i] for i in range(start_frame, end_frame)
            ]
            np_feature_cut = np.array(feature_cut, dtype=np.float32)
            np.save(save_cut_name, np_feature_cut)


def run():
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    gts_bmn = {}
    for item, value in label_files.items():
        label_file = os.path.join(dataset, value)
        gts_data = json.load(open(label_file, 'rb'))
        gts_bmn = gen_gts_for_bmn(gts_data)
    with open(out_dir + '/label.json', 'w', encoding='utf-8') as f:
        data = json.dumps(gts_bmn, indent=4, ensure_ascii=False)
        f.write(data)
    save_feature_to_numpy(gts_bmn, out_dir + '/feature')


if __name__ == '__main__':
    run()
