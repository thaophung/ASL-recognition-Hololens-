"""
1. Change name of videos: 0 -> len(current letter dir)

2. train, val, test = random_choose 
    - Test = 10% of all
    - Val = 10% of all

3. Write txt file for train, val, test

4. Preprocess data




"""
import os
import cv2
import numpy as np
import shutil
import random

# Change video name (rename)
def change_video_name(data_path):
    for dir in os.listdir(data_path):
        count = 0
        letter_dir = os.path.join(data_path, dir)
        for video in os.listdir(letter_dir):
            current_name = os.path.join(letter_dir, video)
            #print(current_name)
            new_name = str(count) + '.mp4'
            new_name = os.path.join(letter_dir, new_name)
            os.rename(current_name, new_name)      # rename the video
            count += 1
    print('Done')

# Generate train, val, test 
def generate_train_val_test(data_path):
    src_dir = os.path.join(data_path, 'video_abc')
    #train_dir = os.path.join(data_path, 'train')
    val_dir = os.path.join(data_path, 'val')
    test_dir = os.path.join(data_path, 'test')
    dirs = os.listdir(src_dir)
    for dir in dirs:
        # number of video in directory 
        num_video = len(os.listdir(os.path.join(src_dir, dir)))
        num_chosen = int(num_video * 0.1)
        taken = []
        for i in range(num_chosen):
            # get a random video and move it for test set
            test_idx = random.randrange(0, num_video-1)
            while test_idx in taken:
                test_idx = random.randrange(0, num_video-1)
            taken.append(test_idx)
            chosen_video = dir + '/'+ str(test_idx) + '.mp4'
            chosen_video_dir = os.path.join(src_dir, chosen_video)
            new_dir = os.path.join(test_dir, chosen_video)
            os.rename(chosen_video_dir, new_dir)

            # get a random video and move it for val set
            val_idx = random.randrange(0, num_video)
            while val_idx in taken:
                val_idx = random.randrange(0, num_video)
            taken.append(val_idx)
            chosen_video = dir + '/' + str(val_idx) + '.mp4'
            chosen_video_dir = os.path.join(src_dir, chosen_video)
            new_dir = os.path.join(val_dir, chosen_video)
            os.rename(chosen_video_dir, new_dir)

    print('Done')

# Write txt files listing all video name for train, val, test
#  in format class_name/video_name
def create_list_txt(path):
    train_txt = os.path.join(path,'trainlist_optical.txt')

    val_txt = os.path.join(path,'vallist_optical.txt')
    #test_txt = os.path.join(path,'testlist.txt')

    train_dir = os.path.join(path, 'train')
    val_dir = os.path.join(path, 'val')
    #test_dir = os.path.join(path, 'test')

    # write trainlist.txt
    with open(train_txt, 'w') as text_file:
        dirs = os.listdir(train_dir)
        for dir in dirs:
            filenames = os.listdir(os.path.join(train_dir, dir))
            for filename in filenames:
                text_file.write(dir + '/' + filename + '\n')

    # write vallist.txt
    with open(val_txt, 'w') as text_file:
        dirs = os.listdir(val_dir)
        for dir in dirs:
            filenames = os.listdir(os.path.join(val_dir, dir))
            for filename in filenames:
                text_file.write(dir + '/' + filename + '\n')

    # write tetslist.txt
#    with open(test_txt, 'w') as text_file:
#        dirs = os.listdir(test_dir)
#        for dir in dirs:
#            filenames = os.listdir(os.path.join(test_dir, dir))
#            for filename in filenames:
#                text_file.write(dir + '/' + filename + '\n')
#
    print('Done')



if __name__ == '__main__':
    #data_path = '/Users/thaophung/workspace/senior_design/dataset/'
    # change_video_name(data_path)
    #generate_train_val_test(data_path)
    data_path = '/Users/thaophung/workspace/senior_design/dataset/optical_flow_dataset_1'
    create_list_txt(data_path)


    
