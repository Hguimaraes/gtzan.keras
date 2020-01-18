from gtzan.data.make_dataset import make_dataset_dl
from gtzan.data.make_dataset import make_dataset_ml
from gtzan.utils import majority_voting
from gtzan.utils import get_genres
from joblib import load
from tensorflow.keras.models import load_model

__all__ = ['AppManager']


class AppManager:
    def __init__(self, args, genres):
        self.args = args
        self.genres = genres

    def run(self):
        if self.args.type == "ml":
            X = make_dataset_ml(self.args)
            pipe = load(self.args.model)
            pred = get_genres(pipe.predict(X)[0], self.genres)
            print("{} is a {} song".format(self.args.song, pred))

        else:
            X = make_dataset_dl(self.args)
            model = load_model(self.args.model)

            preds = model.predict(X)
            votes = majority_voting(preds, self.genres)
            print("{} is a {} song".format(self.args.song, votes[0][0]))
            print("most likely genres are: {}".format(votes[:3]))
