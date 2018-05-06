import numpy as np
import cv2
import random
import scipy.misc
import os
from keras.layers import Dense, Activation, Flatten, Conv2D< MaxPooling2D
from keras.models import Model
from resnet50 import ResNet50

def preprocess_data(video_path):
    """ Get the video data and preprocessing it into stack of frames
    Frame size: (224,224,3)
    Number of frame: 10
    """
    image_size = (224,224,3)
    sequence_length = 10
    all_frames = []
    cap =cv2.VideoCapture(video_path)
    while cap.isOpened():
        succ, frame = cap.read()
        if not succ:
            break
        if frame.any():
            all_frames.append(frame)
    all_frames = np.stack(all_frames, axis=0)
    start_index = random.randrange(len(all_frames) - sequence_length)
    frame_sequence =[]
    for i in range(start_index, start_index + sequence_length - 1):
        frame = scipy.misc.imresize(frame_sequence[i], image_size)
        frame_sequence.append(frame)
    frame_sequence = np.stack(frame_sequence, axis=0)
    cap.release()
    return frame_sequence

N_CLASSES = 26
IMSIZE = (224,224,3)
def finetuned_resnet(weights_dir):
    """ Get the model
    Add on top of ResNet50 2 Fully Connected Layers with Activation Function 'Relu'
    and one more FC layer to output the class of video
    """

    base_model = ResNet50(include_top=False, weights=weights_dir, input_shape=IMSIZE)
    for layer in base_model.layers:
        layer.trainable=False

    model = base_mode.add(Faltten(input_shape=IMSIZE))
    model.add(Dense(2048))
    model.add(Activation('relu'))
    model.add(Dense(1024))
    model.add(Activation('relu'))
    model.add(Dense(1))

    sgd = SGD(lr = 0.003, decay = 1e-6, momentum=0.9, nesterov=True)
    model.load_weights(weights_dir)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model
if __main__ == '__name__':
    # input: a video from hololens
    video_path = ...
    weights_path = ...

    frame_sequence = preprocess_data(video_path)
    model = finetuned_resnet(weights_path)
    im_pred = model.predict(frame_sequence)

    print(im_pred)
