import os
import sys
import subprocess
import shutil
import argparse
import time
from tqdm import tqdm


def extract_frames(source_dir, target_dir):
    source_classes = os.listdir(source_dir)
    source_classes.sort()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # count = 0
    for class_index in tqdm(source_classes, desc='Extracting frames'):
        # if not count % 10:
        #     print("Processing class {}/{}".format(count, len(source_classes)))
        # count += 1
        source_class_dir = os.path.join(source_dir, class_index)
        videos = os.listdir(source_class_dir)
        videos.sort()

        target_class_dir = os.path.join(target_dir, class_index)
        if not os.path.exists(target_class_dir):
            os.makedirs(target_class_dir)

        for each_video in videos:
            source_video_name = os.path.join(source_class_dir, each_video)
            video_prefix = each_video.split('.')[0]
            target_video_frames_folder = os.path.join(target_class_dir, video_prefix)
            if not os.path.exists(target_video_frames_folder):
                os.makedirs(target_video_frames_folder)
            target_frames = os.path.join(target_video_frames_folder, 'frame_%05d.jpg')

            try:
                subprocess.call('ffmpeg -nostats -loglevel 0 -i "%s" -filter:v fps=fps=30 -s 340x256 -q:v 2 "%s"' %
                                (source_video_name, target_frames), shell=True)
                video_frames = os.listdir(target_video_frames_folder)
                video_frames.sort()
                if len(video_frames) == 300:
                    continue
                elif len(video_frames) > 300:
                    for i in range(300, len(video_frames)):
                        os.remove(os.path.join(target_video_frames_folder, video_frames[i]))
                else:
                    last_file = 'frame_%05d.jpg' % (len(video_frames) - 1)
                    last_file = os.path.join(target_video_frames_folder, last_file)
                    for i in range(len(video_frames), 301):
                        new_file = 'frame_%05d.jpg' % i
                        new_file = os.path.join(target_video_frames_folder, new_file)
                        shutil.copyfile(last_file, new_file)
            except:
                print("Error processing video: {}".format(source_video_name))
                continue


def extract_pcm(source_dir, target_dir):
    source_classes = os.listdir(source_dir)
    source_classes.sort()
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for class_index in tqdm(source_classes, desc='Extracting pcm files'):
        source_class_dir = os.path.join(source_dir, class_index)
        videos = os.listdir(source_class_dir)
        videos.sort()
        target_class_dir = os.path.join(target_dir, class_index)
        if not os.path.exists(target_class_dir):
            os.makedirs(target_class_dir)
        for each_video in videos:
            source_video_name = os.path.join(source_class_dir, each_video)
            video_prefix = each_video.split('.')[0]
            target_video_pcm_folder = os.path.join(target_class_dir, video_prefix)
            target_pcm = target_video_pcm_folder + '.pcm'
            try:
                subprocess.call('ffmpeg -y -i %s -acodec pcm_s16le -f s16le -ac 1 -ar 16000 %s -v 0' %
                                (source_video_name, target_pcm))
            except:
                print("Error processing video: {}".format(source_video_name))
                continue


def run(source_dir, target_dir1, target_dir2):
    extract_frames(source_dir, target_dir1)
    extract_pcm(source_dir, target_dir2)


if __name__ == '__main__':
    # python extract_frames.py --source_dir F:/程序代码/PP-TSM/data/tiny-Kinetics-400/train_256 --frames_dir F:/程序代码/PP-TSM/data/tiny-Kinetics-400/frames/train_256_frames --pcm_dir F:/程序代码/PP-TSM/data/tiny-Kinetics-400/pcm/train_256_pcm
    # python extract_frames.py --source_dir F:/程序代码/PP-TSM/data/tiny-Kinetics-400/test_256 --frames_dir F:/程序代码/PP-TSM/data/tiny-Kinetics-400/frames/test_256_frames --pcm_dir F:/程序代码/PP-TSM/data/tiny-Kinetics-400/pcm/test_256_pcm
    parser = argparse.ArgumentParser(
        description='Extract frames of Kinetics400 dataset'
    )
    parser.add_argument('--source_dir', type=str, help='the directory of raw videos')
    parser.add_argument('--frames_dir', type=str,
                        help='the directory which is used to store the extracted frames')
    parser.add_argument('--pcm_dir', type=str,
                        help='the directory which is used to store the extracted pcm files')
    args = parser.parse_args()

    assert args.source_dir, "You must give the source_dir of raw videos!"
    assert args.frames_dir, "You must give the target_dir for storing the extracted frames!"
    assert args.pcm_dir, "You must give the target_dir for storing the extracted pcm files!"

    tic = time.time()
    run(args.source_dir, args.frames_dir, args.pcm_dir)
    print(time.time() - tic)
