import os
import keras.callbacks
from generate_sequence_data import image_from_sequence_generator, sequence_generator, get_data_list
from finetune_resnet import finetuned_resnet
from temporal_CNN import temporal_CNN
from keras.optimizers import SGD
import pickle

N_CLASSES = 26
BatchSize = 32

def fit_model(model, train_data, val_data, weights_dir, input_shape, optical_flow=False):
    try:
        # using sequence or image_from_sequence generator
        if optical_flow:
            train_generator = sequence_generator(train_data, BatchSize, input_shape, N_CLASSES)
            val_generator = sequence_generator(val_data, BatchSize, input_shape, N_CLASSES)
        else:
            train_generator = image_from_sequence_generator(train_data, BatchSize, input_shape, N_CLASSES)
            val_generator = image_from_sequence_generator(val_data, BatchSize, input_shape, N_CLASSES)

        sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        print(model.summary())
        
        print('Starting fitting model')
        while True:
            checkpointer = keras.callbacks.ModelCheckpoint(weights_dir, save_best_only=True, save_weights_only=True)
            earlystopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20,
                    verbose=2, mode='auto')
            tensorboard = keras.callbacks.TensorBoard(log_dir='./logs/try', histogram_freq=0, 
                    write_graph=True, write_images=True)
            history = model.fit_generator(train_generator, steps_per_epoch=200, epochs=2,   #2000
                    validation_data=val_generator, validation_steps=100, verbose=2,
                    callbacks=[checkpointer, tensorboard, earlystopping])
            print(history.history.keys())
            history_file = open('history.pkl', 'w')
            pickle.dumpe(history, history_file)
            


    except KeyboardInterrupt:
        print('Training is interrupted')

if __name__ == '__main__':
    data_dir = '/Users/thaophung/workspace/senior_design/dataset/npy_dataset_1'
    weights_dir = '/Users/thaophung/workspace/senior_design/weights'

    video_dir = '/Users/thaophung/workspace/senior_design/dataset/npy_dataset_1'
    train_data, val_data, class_index = get_data_list(data_dir)
    #print(val_data)
    input_shape = (10, 216, 216, 3)
    weights_dir = os.path.join(weights_dir, 'finetuned_resnet_RGB_65.h5')
    model = finetuned_resnet(include_top=True, weights_dir=weights_dir)
    fit_model(model, train_data, val_data, weights_dir, input_shape)


