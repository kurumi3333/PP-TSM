import csv
import json
import os


def read_labels(file_dir):
    if not os.path.exists(file_dir):
        raise FileNotFoundError("File not found at the given path.")
    subfolder_names = [folder for folder in os.listdir(file_dir) if os.path.isdir(os.path.join(file_dir, folder))]
    # print(subfolder_names)
    result = {}
    for i, subfolder_name in enumerate(subfolder_names):
        result[i + 1] = subfolder_name
        result[subfolder_name] = i+1
    with open('data/tiny-Kinetics-400/label/label.json', 'w') as f:
        json.dump(result, f)
    return subfolder_names


def transform_csv_to_json(csv_file):
    csv_file2 = open(csv_file)
    data = csv.DictReader(csv_file2)
    file_name = csv_file.split('.')[0] + '.json'
    result = {}
    for row in data:
        if row['label'] in result.keys():
            result[row['label']].append(row)
        else:
            result[row['label']] = [row]
    # result = [row for row in data]
    with open(file_name, 'w') as f:
        json.dump(result, f)
    csv_file2.close()
    return result


def gen_video_list(file_dir, output_file):
    with open(output_file, 'w') as f:
        for class_name in read_labels(file_dir):
            class_folder = os.path.join(file_dir, class_name)
            if os.path.isdir(class_folder):
                for video_name in os.listdir(class_folder):
                    f.write(video_name + '\n')


if __name__ == '__main__':
    file_dir_train = "data/tiny-Kinetics-400/train_256"
    file_dir_val = "data/tiny-Kinetics-400/val_256"
    read_labels(file_dir_train)

    csv_files = ['data/tiny-Kinetics-400/label/train_256.csv', 'data/tiny-Kinetics-400/label/val_256.csv']
    for csv_file in csv_files:
        transform_csv_to_json(csv_file)

    output_file_train = "data/tiny-Kinetics-400/train_256/url.list"
    output_file_val = "data/tiny-Kinetics-400/val_256/url.list"
    gen_video_list(file_dir_train, output_file_train)
    gen_video_list(file_dir_train, output_file_val)
