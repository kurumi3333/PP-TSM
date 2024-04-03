import os
import glob
import numpy as np
import paddle
from paddle.io import Dataset
from paddle.vision.transforms import functional as F
from PIL import Image


class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=16):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.data = []
        self.classes = sorted(os.listdir(self.root_dir))
        self.class_to_id = {j: i for i, j in enumerate(self.classes)}
        for class_dir in self.classes:
            class_path = os.path.join(self.root_dir, class_dir)
            for clip_dir in os.listdir(class_path):
                self.data.append((os.path.join(class_path, clip_dir), self.class_to_id[class_dir]))

    def __getitem__(self, index):
        clip_path, class_id = self.data[index]
        frames_path = sorted(glob.glob(os.path.join(clip_path, '*.jpg')))[:self.num_frames]
        frames = [self._load_image(fp) for fp in frames_path]
        if len(frames) < self.num_frames:
            frames = frames + [frames[-1]] * (self.num_frames - len(frames))  # 如果帧数不够，使用最后一帧进行填充
        clip = np.asarray(frames)
        clip = paddle.to_tensor(clip).astype('float32')
        label = paddle.to_tensor(class_id).astype('int64')
        return clip, label

    def __len__(self):
        return len(self.data)

    def _load_image(self, path):
        img = Image.open(path)
        img = F.to_tensor(img)  # 将 PIL Image 转换为 FloatTensor，并且图像的尺寸顺序变为 (C, H, W)
        return img

num_workers = 4
batch_size = 16
dataset_dir = '/path_to_your_dataset'
dataset = VideoDataset(dataset_dir)
data_loader = paddle.io.DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)
