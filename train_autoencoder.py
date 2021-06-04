from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import Callback, LearningRateScheduler
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img
import tensorflow as tf
import argparse
import glob
import os
import shutil
from architecture import *
from utils import *

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training_data', type=str, default='data/processed_cifar/train/', help='Training image path')
    parser.add_argument('--val_data', type=str, default='data/processed_cifar/test/', help='Testing image path')
    parser.add_argument('--aug_data_dir', type=str, default='data/cifar10_aug/', help= 'Path to augment cifar-10 data')
    parser.add_argument('--save_model_path', type=str, default='models/autoencoder_cifar10', help='Path to save the autoencoder trained model')
    parser.add_argument('--img_size', type=int, default=32, help='Input image size')
    parser.add_argument('--num_train_imgs', type=int, default=12500, help='Number of training images')
    parser.add_argument('--num_val_imgs', type=int, default=10000, help='Number of images in testing set')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs for training')
    return parser

# generate same image patch for input and output
def fixed_generator(generator):
    for batch in generator:
        yield (batch, batch)
        
        
def aug_single_img(img_file, cls_name, aug_dir):
    img = load_img(img_file)
    x = img_to_array(img) 
    x = x.reshape((1, ) + x.shape)
    i = 0
    for batch in aug_datagen.flow(x, batch_size=1,
                             save_to_dir= aug_dir + cls_name,
                             save_prefix=cls_name, save_format='png'):
        i += 1
        if i > 3:
            break
            
def augclass(train_dir,aug_dir):
    files = glob.glob(train_dir+'/*/*')
    aug_class = ['airplane','automobile','cat','dog','frog','horse','ship']
    for file in files:
        if file.split('/')[-2] in aug_class:
            cls_name = file.split('/')[-2]
            aug_single_img(file,cls_name, aug_dir)
        else:
            cls_name = file.split('/')[-2]
            shutil.copy(os.path.join(file), os.path.join(aug_dir+cls_name))


if __name__ == '__main__':
    parser = parsing()
    args = parser.parse_args()
    class_name = ["airplane", "automobile", "bird","cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    save_model_name = args.save_model_path.split('/')[-1] + ".h5"
    save_model_path = os.path.join(args.save_model_path, save_model_name)
    out_img_dir = os.path.join(args.save_model_path, 'imgs')
    if not os.path.exists(out_img_dir):
        os.makedirs(out_img_dir)
    
    if not os.path.exists(args.aug_data_dir):
        os.makedirs(args.aug_data_dir)
    for dir_name in class_name:
        os.makedirs(os.path.join('data/cifar10_aug', dir_name))
        
    aug_datagen = ImageDataGenerator(horizontal_flip=True,vertical_flip=True) 
    augclass(args.training_data, args.aug_data_dir)
    
    train_gen = ImageDataGenerator(rescale=1./255)
    val_gen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_gen.flow_from_directory(
                                args.aug_data_dir,
                                target_size=(args.img_size, args.img_size),
                                color_mode='rgb',
                                batch_size=args.batch_size,
                                class_mode=None, shuffle=True)
    
    validation_generator = val_gen.flow_from_directory(
                                    args.val_data,
                                    target_size=(args.img_size, args.img_size),
                                    color_mode='rgb',
                                    batch_size=args.batch_size,
                                    class_mode=None, shuffle=True)
    
    
    input_img = Input(shape=(args.img_size, args.img_size,3))
    encoded = Net_Encoder(input_img)
    decoded = Net_Decoder(encoded)
    ae_cnn = Model(input_img, decoded)
    print(ae_cnn.summary())
    
    optimizer = Adam(lr=0.00001)
    ae_cnn.compile(optimizer=optimizer, loss='mse')
    
    save_imgs = SaveOutputImages(validation_generator, out_img_dir)
    
    history = ae_cnn.fit_generator(
                    fixed_generator(train_generator),
                    steps_per_epoch=args.num_train_imgs// args.batch_size,
                    epochs = args.num_epochs,
                    validation_data = fixed_generator(validation_generator),
                    validation_steps = args.num_val_imgs// args.batch_size,
                    callbacks= [save_imgs, LearningRateScheduler(lr_schedule_ae)])
    
    
    ae_cnn.save(save_model_path)
    
    PlotTrainValLoss(history, args.save_model_path, args.num_epochs)
    
    # Test random images
    test_x = validation_generator.next()
    decoded_imgs = ae_cnn.predict(test_x)
    VisualizeAE(test_x, decoded_imgs, args.save_model_path, args.num_epochs)