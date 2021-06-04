import os
import argparse
import shutil

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data/cifar/train', help='Data directory path for preparing data')
    parser.add_argument('--output_dir', type=str, default='data/cifar10_processed/train', help='Processed data directory path')
    parser.add_argument('--set_num_classes', dest='classes_list', action='store_true', help='This allows you to choose default cifat10 dataset')
    parser.add_argument('--set_training_classes', type= str, default='cifar10_train_labels.txt', help='Path to the text config file tto set number of samples for each classes')
    parser.set_defaults(classes_list=False)
    return parser



# This function read the config text file containing number of images for each labels
# and return dictionary with key pair as 'class name: number of images'
def ParseSetTrainingClasses(txtfile):
    class_num_dict = {}
    with open(txtfile) as f:
        lines = f.read().splitlines()
        for line in lines:
            values = line.split(' ')
            class_label = values[0]
            max_label_num = int(values[1])
            class_num_dict[class_label] = max_label_num
            
    return class_num_dict

if __name__ =='__main__':
    parser = parsing()
    args = parser.parse_args()
    
    if args.classes_list:
        if os.path.isfile(args.set_training_classes):
            class_num_dict = ParseSetTrainingClasses(args.set_training_classes)
            print("Training data config: {}".format(class_num_dict))
        else:
            print('Default training config for cifar10')
            quit()
            
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
        
    # file name format is "number_classname.png"
    for file in os.scandir(args.data_dir):
        if file.is_file() and file.name.endswith('.png'):
            file_name = os.path.splitext(file.name)[0]
            class_name = file_name.split('_')[-1]
            output_dir = os.path.join(args.output_dir, class_name)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
                
            if args.classes_list:
                if class_num_dict[class_name] > 0:
                    shutil.copy(os.path.join(args.data_dir, file.name), output_dir)
                    class_num_dict[class_name] = class_num_dict[class_name] - 1
            else:
                shutil.copy(os.path.join(args.data_dir, file.name), output_dir)
                

            
            