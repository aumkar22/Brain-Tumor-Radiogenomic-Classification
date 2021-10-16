# Brain-Tumor-Radiogenomic-Classification
Repository for kaggle competition, "RSNA-MICCAI Brain Tumor Radiogenomic Classification" https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/discussion/262046

## Running the code

Install dependencies as:

```
pip install -r requirements.txt
```

You can run this code from command line as

```
python train.py -m resnet101
```

Arguments are optional, if not provided, baseline resnet model will be used for training.

## Repository structure
```
Brain-Tumor-Radiogenomic-Classification
│   README.md
|   train.py                                              // Script to train the models
|
└───src
|    └───scripts                                          // Scripts for data loading and batch generation
|    |    rsna_load.py
|    |    rsna_generator.py
|    |    ...
|    └───preprocessing                                    // Preprocessing script
|    |    data_preprocess.py
|    └───models                                           // Scripts for baseline ResNet model and other transfer learning models
|    |    base_model.py
|    |    tl_model.py
|    |    ...
|    └───util                                             // Utility scripts
|    |    definitions.py
|    |    folder_check.py
|    |    └───config
|    |    |    resnet_params.yaml
└───data                                                  // RSNA-MICCAI Brain Tumor Radiogenomic Classification data
|     └───models                                          // Model logs and save
|     |      └───resnet101 
|     |      └───resnext50                                                 
|     |      └───...
|     └───train                                           // Train data
|     |      └───00000 
|     |      |      └───FLAIR
|     |      |      |    Image-1.dcm
|     |      |      |    Image-2.dcm
|     |      |      |    ...
|     |      |      └───T1w
|     |      |      └───T1wCE
|     |      |      └───T2w
|     |      └───00002 
|     |      |      └───FLAIR
|     |      |      └───T1w
|     |      |      └───T1wCE
|     |      |      └───T2w
|     |      └───...
|     └───test                                             // Test data
|     |      └───00001 
|     |      |      └───FLAIR
|     |      |      |    Image-1.dcm
|     |      |      |    Image-2.dcm
|     |      |      |    ...
|     |      |      └───T1w
|     |      |      └───T1wCE
|     |      |      └───T2w
|     |      └───00013 
|     |      |      └───FLAIR
|     |      |      └───T1w
|     |      |      └───T1wCE
|     |      |      └───T2w
|     |      └───...
|     └───train_npy                                        // Preprocessed train data
|     |    00000.npz
|     |    00002.npz
|     |    ...
|     └───validation_npy                                   // Preprocessed validation data
|     |    00003.npz
|     |    00009.npz
|     |    ...
|     └───test_npy                                         // Preprocessed test data
|     |    00001.npz
|     |    00013.npz
|     |    ...
|     |   train_labels.csv                                 // File containing the target MGMT_value for each patient in the train data
      
```

## MRI Slice

### Original

![](https://i.imgur.com/4LDS5XJ.png)

### Preprocessed

![](https://i.imgur.com/0DChBzf.png)

### Cropped and resized

![](https://i.imgur.com/GCp8TpG.png)
