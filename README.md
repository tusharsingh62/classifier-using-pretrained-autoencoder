# classifier-using-pretrained-autoencoder

### Tested on docker container

- Build docker image from Dockerfile
```
docker build -t cifar .
```

- Create docker container based on above docker image
```
docker run --gpus 0 -it -v $(pwd):/mnt -p 8080:8080 cifar
```

- Enter docker container and follow the steps to reproduce the experiments results
```
docker exec -it {container_id} /bin/bash
```

- Current directory is mounted on ***/mnt*** path inside the docker container
- Go to /mnt directory inside the docker container


### Download CIFAR-10 dataset
```
cd data/

sudo wget http://pjreddie.com/media/files/cifar.tgz

tar xzf cifar.tgz

rm cifar.tgz
```

### Process CIFAR-10 dataset and prepare train, test dataset according to the cifar10_train_labels.txt file
```
cd ../ # Move to the root folder

python prepare_cifar10.py --images_dir data/cifar/train/ --out_dir data/processed_cifar/train/ --set_num_classes

python prepare_cifar10.py --images_dir data/cifar/test/ --out_dir data/processed_cifar/test/

```

### Distribution of training dataset after processing the cifar-10
```
Training data config: {'ship': 716, 'airplane': 714, 'deer': 2500, 'bird': 2500, 'horse': 714, 'cat': 714, 'truck': 2500, 'automobile': 714, 'dog': 714, 'frog': 714}
```

### Handling Imbalance dataset
- Data Augmentation of minority classes
- Setup class weights in the loss function

### Data Augmentation and Train the autoencoder

```
python train_autoencoder.py
```
- Please check the default parameters for above autoencoder training script
- It will create another **data/cifar10_aug** cifar-10 data directory after augmentation
- Also it start training the autoencoder (unsupervised learning) on augmented cifar-10 dataset

### Train Classifier

### Experiment-1
- Optimizer--> SGD
- Xavier Initialization
- Data Augmentation
- Others are default parameters

```
python train_classifier.py --add_fc_layers --train_from_scratch
```

### Experiment-2
- Optimizer--> SGD
- Autoencoder pretrained Initialization
- Data Augmentation
- Others are default parameters
```
python train_classifier.py --add_fc_layers
```

### Experiment-3
- Optimizer--> SGD
- Autoencoder pretrained Initialization
- Weight balance for each classes in the loss function
- Others are default parameters
```
python train_classifier.py --add_fc_layers --balance_weights
```

### Experiment-4
- Optimizer--> SGD
- Xavier Initialization
- Weight balance for each classes in the loss function
- Others are default parameters

```
python train_classifier.py --add_fc_layers --balance_weights --train_from_scratch
```

### Evalution

```
python eval.py --model_path {} --data_dir {}
```

## Experiment Results

### AutoEncoder Reconstruction

![AutoEncoder Reconstruction](/exp_results/reconstruct_epoch50.png)

### AutoEncoder Training 

![AutoEncoder Training](/exp_results/autoencoder_training.png)

### All experiments results

![Experiment Results](/exp_results/exp_table_results.png)

### Data Augmentation SGD with prerained auto encoder initialization

![Confusion Matrix](/exp_results/classification_cifar10_confusion_matrix.jpg)

