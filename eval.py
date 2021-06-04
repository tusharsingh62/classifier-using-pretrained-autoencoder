import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from utils import CalculateConfusionMatrix, GetCifar10Mean, GetCifar10STD
import argparse

def parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/classification_cifar10/classification_cifar10.h5', help='Path for trained classification model')
    parser.add_argument('--data_dir', type=str, default='data/processed_cifar/test', help='Path for testing images')
    return parser

if __name__ == '__main__':
    parser = parsing()
    args = parser.parse_args()
    
    confusion_matrix_img = os.path.splitext(args.model_path)[0] + '_cm.jpg'
    
    dic_cls = {0:"airplane", 1:"automobile", 2:"bird", 3:"cat", 4:"deer", 5:"dog", 6:"frog", 7:"horse", 8:"ship", 9:"truck"}
    cls_names = ["airplane", "automobile", "bird","cat", "deer", "dog", "frog", "horse", "ship", "truck"]

    model = load_model(args.model_path)
    
    pred = []
    true_cls = []
    pred_cls = []
    for cls_dir in os.listdir(args.data_dir):
        cls_path = os.path.join(args.data_dir, cls_dir)
        if os.path.isdir(cls_path):
            cls_name = str(cls_dir)
            for files in os.scandir(cls_path):
                if files.is_file() and (files.name.endswith('.png') or files.name.endswith('.jpg')):
                    img_path = os.path.join(cls_path, files.name)
                    test_img = image.load_img(img_path)
                    test_img = image.img_to_array(test_img)
                    test_img = np.expand_dims(test_img, axis=0)*(1./255)
                    test_img = (test_img - GetCifar10Mean()) / GetCifar10STD()
                    
                    preds = model.predict(test_img)
                    pred_id = np.argmax(preds)
                    if dic_cls[pred_id] == cls_name:
                        pred.append(1)
                    else:
                        pred.append(0)
                        
                    true_cls.append(cls_name)
                    pred_cls.append(dic_cls[pred_id])
    acc = pred.count(1) / len(pred)
    print("Accuracy of the classification model with cifar10 test set is: ".format(acc))
    CalculateConfusionMatrix(true_cls, pred_cls, cls_names, confusion_matrix_img, acc)
                    
                        
                        
        


