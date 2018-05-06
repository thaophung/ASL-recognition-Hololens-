import numpy as np
import scipy.misc
import os, cv2, random
import shutil

def combine_list_txt(data_path):
    trainlisttxt = 'trainlist.txt'
    vallisttxt = 'vallist.txt'
    #testlisttxt = 'testlist.txt'

    trainlist = []
    txt_path = os.path.join(data_path, trainlisttxt)
    with open(txt_path) as fo:
        for line in fo:
            trainlist.append(line[:line.rfind(' ')])

    vallist = []
    txt_path = os.path.join(data_path, vallisttxt)
    with open(txt_path) as fo:
        for line in fo:
            vallist.append(line[:line.rfind(' ')])

    return trainlist, vallist


def regenerate_data(data_path):
    sequence_length = 10
    image_size = (224,224,3)

    dest_dir = os.path.join(data_path, 'npy_dataset_2')
    # generate sequence for optical flow
    preprocessing(data_path, dest_dir, sequence_length, image_size, overwrite=True,
            normalization=False, mean_subtraction=False, horizontal_flip=False,
            random_crop=False, consistent=False, continuous_seq=True)

    # compute optical flow data


def preprocessing(data_path, dest_dir, seq_len, img_size, overwrite=False, 
        normalization=False, mean_subtraction=False, horizontal_flip=False, 
        random_crop=False, consistent=False, continuous_seq=True):
    '''
    Extract video data to sequence of fixed length, and save it in npy file
    :param list_dir
    :param data_dir
    :param seq_len
    :param img_size:
    :param overwrite: 
    :param normalizaation: normalize to (0,1)
    :param mean_subtraction: subtract mean of RGB channels
    :param horizontal_flip: add random noise to sequence data
    :param random_crop: cropping using random location
    :param consistent: whether horizontal flip, random crop is consistent in sequence
    :param continuous_seq: whether frames extracted are continuous
    :return:
    '''
    
    #write a txt file to keep parameter inforamtion
    txt_file = os.path.join(dest_dir,'parameters.txt')
    with open(txt_file,'w') as fo:
        fo.write('seq_len: ' + str(seq_len) +
                '\noverwrite: ' + str(overwrite) +
                '\nnormalization: ' + str(normalization) + 
                '\nmean_subtraction: ' + str(mean_subtraction) + 
                '\nhorizontal_flip: ' + str(horizontal_flip) + 
                '\nrandom_crop: ' + str(random_crop) + 
                '\nconsistent: ' + str(consistent) + 
                '\ncontinuous_seq: ' + str(continuous_seq))
                

    trainlist, vallist = combine_list_txt(data_path)
    train_src = os.path.join(data_path, 'train')
    val_src = os.path.join(data_path, 'val')

    train_dir = os.path.join(dest_dir, 'train')
    val_dir = os.path.join(dest_dir, 'val')
    #os.mkdir(train_dir)
    #os.mkdir(val_dir)
    if mean_subtraction:
        mean = calc_mean(UCF_dir, img_size).astype(dtype='float16')
        np.save(os.path.join(dest_dir, 'mean.npy'), mean)
    else:
        mean = None

    print('Preprocessing ASL data ....')
    for clip_list, sub_dir in [(trainlist, train_dir)]: #, (vallist, val_dir)]:
        for clip in clip_list:
            clip_name = os.path.basename(clip)
            clip_category = os.path.dirname(clip)
            category_dir = os.path.join(sub_dir, clip_category)
            if sub_dir == train_dir:
                src_dir = os.path.join(train_src, clip)
            else:
                src_dir = os.path.join(val_src, clip)
            dst_dir = os.path.join(category_dir, clip_name)

            if not os.path.exists(category_dir):
                os.mkdir(category_dir)
            process_clip(clip_category, src_dir, dst_dir, seq_len, img_size, mean=mean, 
                    normalization=normalization, horizontal_flip=horizontal_flip,
                    random_crop=random_crop, consistent=consistent,
                    continuous_seq=continuous_seq)
    print("Processing done...")

# down sample image resolution to 216*216, and make sequence length 10
def process_clip(clip_category, src_dir, dst_dir, seq_len, img_size, mean=False, normalization=False,
        horizontal_flip=False, random_crop=False, consistent=False, continuous_seq=False):
    all_frames = []
    cap = cv2.VideoCapture(src_dir)
    while cap.isOpened():
        succ, frame = cap.read()
        if not succ:
            break
        # append frame that is not all zeros
        if frame.any():
            all_frames.append(frame)
    
    clip_length = len(all_frames)
    
    # save all frames
    if seq_len is None or clip_length <= 10 or clip_category =='j' or clip_category == 'z':
        #print('normal ' + src_dir)
        print(src_dir)
        all_frames = np.stack(all_frames, axis=0)
        dst_dir = os.path.splitext(dst_dir)[0] + '.npy'
        np.save(dst_dir, all_frames)
    else:
        step_size = int(clip_length / (seq_len))
        frame_sequence = []
        # select random first frame index for continous sequence
        if continuous_seq:
            start_index = random.randrange(clip_length-seq_len + 1)
        # choose whether to flip or not for all frames
        if not horizontal_flip:
            flip = False
        elif horizontal_flip and consistent:
            flip = random.randrange(2) == 1
        if not random_crop:
            x, y = None, None
        xy_set = False
        for i in range(seq_len):
            if continuous_seq:
                index = start_index + i
            else:
                index = i * step_size + random.randrange(step_size)
            frame = all_frames[index]
            # compute flip for each frame 
            if horizontal_flip and not consistent:
                flip = random.randrange(2) == 1
            if random_crop and consistent and not xy_set:
                x = random.randrange(frame.shape[0] - img_size[0])
                y = random.randrange(frame.shape[1] - img_size[1])
                xy_set = True
            elif random_crop and not consistent:
                x = random.randrange(frame.shape[0] - img_size[0])
                y = random.randrange(frame.shape[1] - img_size[1])
            frame = process_frame(frame, img_size, x, y, mean=mean, 
                    normalization=normalization, flip=flip, 
                    random_crop=random_crop)
            frame_sequence.append(frame)
        frame_sequence = np.stack(frame_sequence, axis=0)
        dst_dis = os.path.splitext(dst_dir)[0] + '.npy'
        np.save(dst_dir, frame_sequence)
    cap.release()

def process_frame(frame, img_size, x, y, mean=None, normalization=True, flip=True, 
        random_crop=False):
    if not random_crop:
        frame = scipy.misc.imresize(frame, img_size)
    else:
        frame = frame[x:x+img_size[0], y:y+img_size[1],:]
    # flip horizontally
    if flip:
        frame = frame[:, ::-1, :]
    frame = frame.astype(dtype='float16')
    if mean is not None:
        frame -=mean
    if normalization:
        frame /= 255

    return frame 

if __name__ == '__main__':
    '''
        extract frames from videos as npy files
    '''

    #sequence_length = None
    #image_size = (216,216,3)

    data_path = '/Users/thaophung/workspace/senior_design/dataset'

    regenerate_data(data_path)
