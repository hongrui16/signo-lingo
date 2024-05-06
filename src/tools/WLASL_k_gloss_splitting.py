"""
The script allows to divide the WLASL dataset into sub-datasets. The division
is made according to the order indicated in the JSON file. This file is made
available by the authors of WLASL dataset.

Usage: python k_gloss_splitting.py param1 param2
 - param1: path to the full dataset (e.g. ./WLASL_full/)
 - param2: number of glosses to be considered for the split (e.g. 2000)
"""
import json
import os
import shutil
import sys

import cv2
from tqdm import tqdm


def read_json(file_path):
    with open(file_path) as f:
        wlasl_json = json.load(f)
    return wlasl_json



def get_meta_data(file_path):
    video_cap = cv2.VideoCapture(file_path)
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps
    video_cap.release()

    return duration, frame_count, fps


def dump_anno():
    # global variables
    PATH_JSON = '/scratch/rhong5/dataset/signLanguage/WLASL/WLASL_v0.3.json'
    n_glosses = 100
    n_glosses = 300


    root_dir = '/scratch/rhong5/dataset/signLanguage/WLASL'
    sub_dataset_root_dir = f'{root_dir}/WLASL{str(n_glosses)}'
    os.makedirs(sub_dataset_root_dir, exist_ok=True)

    splits = ['train', 'val', 'test']
    for split in splits:
        os.makedirs(f'{sub_dataset_root_dir}/{split}/anno', exist_ok=True)

    try:
        
        if not 1 <= n_glosses <= 2000:
            raise ValueError('\nInsert an integer: 1~2000')

        wlasl_json = read_json(PATH_JSON)
        label_set = set()
        train_group = {}
        val_group = {}
        test_group = {}
        for k, gloss in tqdm(enumerate(wlasl_json)):  # iterate through each gloss
            if k < n_glosses:
                video_infos = gloss['instances']  # get all videos as array
                gloss_name = gloss['gloss']
                print(f'processing {k}/{n_glosses}: {gloss_name}')
                label_set.add(gloss_name)
                for v_info in video_infos:
                    video_id = v_info['video_id']
                    split = v_info['split']  # get destination dir
                    ## add gloss_name to the v_info dict
                    v_info['gloss'] = gloss_name
                    anno_dir = f'{sub_dataset_root_dir}/{split}/anno'
        
                    # # dump the video info to the json file, save to anno_dir, name as [video_id].json
                    # anno_json_filepath = f'{anno_dir}/{video_id}.json'
                    # if os.path.exists(anno_json_filepath):
                    #     os.remove(anno_json_filepath)
                    # with open(anno_json_filepath, 'w') as f:
                    #     json.dump(v_info, f)

                    if split == 'train':
                        train_group[video_id] = v_info
                    elif split == 'val':
                        val_group[video_id] = v_info
                    elif split == 'test':
                        test_group[video_id] = v_info
                    
            else:
                break
        train_json_filepath = f'{sub_dataset_root_dir}/train.json'
        if os.path.exists(train_json_filepath):
            os.remove(train_json_filepath)
        with open(train_json_filepath, 'w') as f:
            json.dump(train_group, f)
        
        val_json_filepath = f'{sub_dataset_root_dir}/val.json'
        if os.path.exists(val_json_filepath):
            os.remove(val_json_filepath)
        with open(val_json_filepath, 'w') as f: 
            json.dump(val_group, f)

        test_json_filepath = f'{sub_dataset_root_dir}/test.json'
        if os.path.exists(test_json_filepath):
            os.remove(test_json_filepath)
        with open(test_json_filepath, 'w') as f:
            json.dump(test_group, f)



        ### dump the label_set to a txt file, save to sub_dataset_root_dir, name as label_set.txt
        label_set_filepath = f'{sub_dataset_root_dir}/label_set.txt'
        if os.path.exists(label_set_filepath):
            os.remove(label_set_filepath)
        with open(label_set_filepath, 'w') as f:
            for item in label_set:
                f.write("%s\n" % item)

        print('\n[log] > DONE!')

    except ValueError:
        print('Insert an integer: 1~2000')



def get_video_info():
    # annno_dir = '/scratch/rhong5/dataset/signLanguage/WLASL/WLASL_300/train/anno'
    annno_dir = '/scratch/rhong5/dataset/signLanguage/WLASL/WLASL_100/train/anno'
    video_root_dir = '/scratch/pfayyazs/datasets/WLASL2000'
    json_files = [f for f in os.listdir(annno_dir) if f.endswith('.json')]
    frame_count_list = []
    for json_file in json_files:
        json_file_path = os.path.join(annno_dir, json_file)
        with open(json_file_path) as f:
            video_info = json.load(f)
            frame_end = video_info['frame_end']
            frame_start = video_info['frame_start']
            video_id = video_info['video_id']
            fps = video_info['fps']

            if frame_end == -1:
                video_path = f'{video_root_dir}/{video_id}.mp4'
                duration, frame_count, fps = get_meta_data(video_path)
            else:
                frame_count = frame_end - frame_start
            print(f'{video_id}.mp4 frame_count: {frame_count}, fps: {fps}')
            frame_count_list.append(frame_count)
        ## close json file
        f.close()

    ## calculate the average frame count
    avg_frame_count = sum(frame_count_list) / len(frame_count_list)
    print(f'average frame count: {avg_frame_count}')

    ## calculate the max frame count
    max_frame_count = max(frame_count_list)
    print(f'max frame count: {max_frame_count}')

    ## calculate the min frame count
    min_frame_count = min(frame_count_list)
    print(f'min frame count: {min_frame_count}')

    ## calculate the median frame count
    frame_count_list.sort()
    n = len(frame_count_list)
    if n % 2 == 0:
        median_frame_count = (frame_count_list[n//2 - 1] + frame_count_list[n//2]) / 2
    else:
        median_frame_count = frame_count_list[n//2]
    print(f'median frame count: {median_frame_count}')
    





if __name__ == '__main__':
    dump_anno()
    # get_video_info()