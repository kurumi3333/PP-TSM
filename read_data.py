import os

def read_labels(file_dir):
    if not os.path.exists(file_dir):
        raise FileNotFoundError("File not found at the given path.")
    subfolder_names = [folder for folder in os.listdir(file_dir) if os.path.isdir(os.path.join(file_dir, folder))]
    print(subfolder_names)
    return subfolder_names


if __name__ == '__main__':
    file_dir = "data/tiny-Kinetics-400/train_256"
    read_labels(file_dir)


