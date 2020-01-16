## gtzan.keras

>  Music Genre classification using Convolutional Neural Networks. Implemented in Tensorflow 2.0 using the Keras API

### Overview

*tl;dr*: Compare the classic approach of extract features and use a classifier (e.g SVM) with the modern approach of using CNNs on a representation of the audio (Melspectrogram). You can see both approaches on the **nbs** folder in the Jupyter notebooks. 

For the deep learning approach:

1. Shuffle the input and split into train and test (70%/30%)
2. Read the audios as melspectrograms, spliting then into 1.5s windows with 50% overlaping resulting in a dataset with shape (samples x time x frequency x channels)
3. Train the CNN and test on test set using a Majority Voting approach

### Results

To compare the result across multiple architectures, we have took two approaches for this problem: One using the classic approach of extracting features and then using a classifier. The second approach, wich is implemented on the src file here is a Deep Learning approach feeding a CNN with a melspectrogram.

You can check in the nbs folder on how we extracted the features, but here are the current results for a k-fold (k = 5):

| Model | Acc |
|-------|-----|
| Decision Tree | 0.5160 |
| Logistic Regression | 0.7640 |
| ElasticNet | 0.6880 |
| Random Forest | 0.6760 |
| **SVM (RBF)** | **0.7880** |

For the deep learning approach we have tested two models: CNN 2D and a VGG16-like model. Reading the file as melspectrograms and split the 30s into 3s files with 50% overlaping, using a training split of 70% for train and 30% test. The process was executed 3x to ensure it wasn't a luck split.

| Model | Acc |
|-------|-----|
| CNN 2D | 0.832 |

![alt text](./data/assets/losscnn2dgtzan.png "Loss and accuracy of the CNN model")
![alt text](./data/assets/cmcnngtzan.png "Confusion Matrix of the CNN model")

### Dataset

And how to get the dataset?

1. Download the GTZAN dataset [here](http://opihi.cs.uvic.ca/sound/genres.tar.gz)

Extract the file in the **data** folder of this project. The structure should look like this:

```bash
├── data/
   ├── genres
      ├── blues
      ├── classical
      ├── country
      .
      .
      .
      ├── rock
```

### How to run

The models are provided as **.joblib** or **.h5** files in the *models* folder. You just need to use it on your custom file as described bellow.

If you want to run the training process yourself, you need to run the provided notebooks in *nbs* folder.

To apply the model on a test file, you need to run:

```bash
$ cd src/
$ python app.py -t MODEL_TYPE -m ../models/PATH_TO_MODEL -s PATH_TO_SONG
```

Where MODEL_TYPE = [ml, dl] for classical machine learning approach and for a deep learning approach, respectivly. 
