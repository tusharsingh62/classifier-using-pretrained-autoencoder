import os
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras import regularizers
import argparse
import glob
from sklearn.utils import class_weight
from architecture import *
from utils import *

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data', type=str, default='data/processed_cifar/train/', help='Training image path')
    parser.add_argument('--aug_data_dir', type=str, default='data/cifar10_aug/', help= 'Path to augment cifar-10 data')
    parser.add_argument('--val_data', type=str, default='data/processed_cifar/test/', help='Testing image path')
    parser.add_argument('--save_model_path', type=str, default='models/classification_cifar10',help='Path to save the classification trained model')
    parser.add_argument('--trainer_ae_path', type=str, default='models/autoencoder_cifar10/autoencoder_cifar10.h5', help='Path of autoencoder trained model')
    parser.add_argument('--img_size', type=int, default=32, help='Size of input image')
    parser.add_argument('--num_val_imgs', type=int, default=10000, help='Number of images in testing set')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--add_fc_layers', action='store_true', help = 'addition for classification')
    parser.add_argument('--train_from_scratch', action='store_true', help='training from scratch')
    parser.add_argument('--freeze_base_arch', action='store_true')
    parser.add_argument('--balance_weights', action='store_true')
    return parser

if __name__ == '__main__':
    parser = parsing()
    args = parser.parse_args()
    save_model_name = args.save_model_path.split('/')[-1] + '.h5'
    save_model_path = os.path.join(args.save_model_path, save_model_name)
    if not os.path.exists(args.save_model_path):
        os.makedirs(args.save_model_path)
        
    unique_classes = ["airplane", "automobile", "bird","cat", "deer", "dog", "frog", "horse", "ship", "truck"]
        
    img_files = glob.glob(args.training_data+'*/*')
    class_list = []
    for file in img_files:
        class_list.append(file.split('/')[-2])
    
    class_weights = class_weight.compute_class_weight('balanced',
                                                 unique_classes,
                                                 class_list)   
        
    if args.train_from_scratch:
        print('Training classification model from scratch...')
        input_img = Input(shape=(args.img_size, args.img_size, 3))
        base_model = Net_Encoder(input_img)
        if args.add_fc_layers:
            net = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001), name='class_dense')(base_model)
        predict = Dense(args.num_classes, activation='softmax')(net)
        model = Model(inputs=input_img, outputs=predict)
        
    else:
        pretrained_ae = load_model(args.trainer_ae_path)
        base_model = Model(inputs=pretrained_ae.input, outputs=pretrained_ae.get_layer('latent_feats').output)
        print('Pretrained autoencoder model loaded....')
        net = base_model.output
        if args.add_fc_layers:
            x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.0001),name='class_dense')(net)
            predict = Dense(args.num_classes, activation='softmax')(x)
        
        model = Model(inputs=base_model.input, outputs=predict)
        
        if args.freeze_base_arch:
            print('Freezed feature extractor....')
            for layer in base_mode.layers:
                layer.trainable = False
    
    print(model.summary())
    
    optim = SGD(lr=0.001, momentum=0.9, decay=1e-6)
    if args.balance_weights:
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['acc'], class_weight= class_weights)
        print('Loss function Weight Balance for Handling imbalance data')
    else:
        model.compile(optimizer=optim, loss='categorical_crossentropy', metrics=['acc'])
        print('Data Augmentation for Handling imbalance data')
        
        
        
    
    train_gen = ImageDataGenerator(
                rescale=1. / 255,
                featurewise_center=True,
                featurewise_std_normalization=True,
                horizontal_flip=True,
                vertical_flip=True,
                rotation_range=15,
                width_shift_range=0.1,
                height_shift_range=0.1)
    
    train_gen.mean = GetCifar10Mean()
    train_gen.std = GetCifar10STD()
    
    val_gen = ImageDataGenerator(rescale=1./ 255,
                featurewise_center=True,
                featurewise_std_normalization=True)
    val_gen.mean = GetCifar10Mean()
    val_gen.std = GetCifar10STD()
    
    if args.balance_weights:
        num_train_imgs = len(glob.glob(args.training_data+'*/*'))
    
        train_generator = train_gen.flow_from_directory(
                        args.training_data,
                        target_size=(args.img_size, args.img_size),
                        color_mode='rgb',
                        batch_size = args.batch_size,
                        class_mode='categorical')
    else:
        num_train_imgs = len(glob.glob(args.aug_data_dir+'*/*'))
        train_generator = train_gen.flow_from_directory(
                        args.aug_data_dir,
                        target_size=(args.img_size, args.img_size),
                        color_mode='rgb',
                        batch_size = args.batch_size,
                        class_mode='categorical')
        
    
    val_generator = val_gen.flow_from_directory(
                    args.val_data,
                    target_size=(args.img_size, args.img_size),
                    color_mode='rgb',
                    batch_size = args.batch_size,
                    class_mode='categorical')
    
    history = model.fit_generator(
                train_generator,
                steps_per_epoch= num_train_imgs// args.batch_size,
                epochs = args.num_epochs,
                validation_data = val_generator,
                validation_steps = args.num_val_imgs// args.batch_size,
                shuffle=True, workers=8, verbose=1,
                callbacks=[LearningRateScheduler(lr_schedule)])
    model.save(save_model_path)
    
    PlotTrainValAccuracy(history, args.save_model_path, args.num_epochs)
    PlotTrainValLoss(history, args.save_model_path, args.num_epochs)
                    
    
    
    
    
    
            



    