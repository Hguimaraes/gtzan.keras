from gtzan.data.make_dataset import make_dataset_dl
from gtzan.data.make_dataset import make_dataset_ml
from gtzan.utils import majority_voting
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
			pred = pipe.predict(X)
			print("{} is a {} song".format(self.args.song, pred))

		else:
			X = make_dataset_dl(self.args, self.genres)
			model = load_model(self.args.model)

			preds = model.predict(X)
			votes = majority_voting(preds)
			print("{} is a {} song".format(self.args.song, votes[0]))
			print("other possible genres are: {}".format(votes[1:3]))
