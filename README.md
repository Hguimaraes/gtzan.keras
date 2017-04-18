## gtzan.keras

>  Music Genre classification using Convolutional Neural Networks. Implemented in Keras

### Dependencies

* Numpy
* Librosa
* Keras
* Tensorflow
* Scikit-learn

### Dataset

And how to get the dataset?

1. Download the GTZAN dataset [here](http://opihi.cs.uvic.ca/sound/genres.tar.gz)

Extract the file in the dataset folder of this project. The structure should look like this:

```bash
├── dataset/
   ├── GTZAN
      ├── blues
      ├── classical
      ├── country
      .
      .
      .
      ├── rock
```

### How to run

1. To run the CNN on the MelSpectrogram

```bash
$ cd src/
$ python gtzan.py SPECT
```

2. To run the CNN on the MFCC

```bash
$ cd src/
$ python gtzan.py MFCC
```

You can tune some parameters in the gtzan.py
The models can be found in the models.py file. Tune as you want.