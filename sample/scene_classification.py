import sys
from os import listdir, makedirs
from os.path import join, exists, splitext
import numpy as np
from sklearn.utils import shuffle 
from sklearn.externals import joblib
from sklearn import linear_model
import pickle

# Configure
train_by_all = False
n_trains = 200

# Read command line arguments
argv = sys.argv
argc = len(argv)
if argc < 2:
	print "Not enough arguments."
	quit(0)

if argc >=3:
	save_model = True
	model_save_dir = argv[2]
else:
	save_model = False

# Prepare directory
feature_dir = argv[1]
if save_model and not exists(model_save_dir):
	makedirs(model_save_dir)

# Iteration to load features
filenames = listdir(feature_dir)
class_id = 0
class_names = []
for filename in filenames:
	filepath = join(feature_dir, filename)
	if not exists(filepath):
		continue

	# Get class label
	class_name, _ = splitext(filename)

	# Load and shuffle features
	features = shuffle(np.load(filepath))

	# Construct train set
	if train_by_all:
		n_trains = features.shape[0]

	## Features
	try:
		train_feat = np.vstack((train_feat, features[0:n_trains,:]))
	except:
		train_feat = np.atleast_2d(features[0:n_trains,:])

	## Labels
	train_label_vec = np.ones(n_trains) * class_id
	try:
		train_label = np.hstack((train_label, train_label_vec))
	except:
		train_label = train_label_vec

	# Construct test set
	if train_by_all:
		n_tests = 100
	else:
		n_tests = features.shape[0] - n_trains
	ind_test = features.shape[0] - n_tests

	# Features
	try:
		test_feat = np.vstack((test_feat, features[ind_test:]))
	except:
		test_feat = np.atleast_2d(features[ind_test:])

	# Labels
	test_label_vec = np.ones(n_tests) * class_id
	try:
		test_label = np.hstack((test_label, test_label_vec))
	except:
		test_label = test_label_vec

	# Preserve class label and its id
	class_names.append(class_name)
	class_id += 1

# Shuffle train data
train_feat, train_label = shuffle(train_feat, train_label)

print "Train Classifier..."
clf = linear_model.SGDClassifier(average=50, n_iter=20)
clf.fit(train_feat, train_label)

print "Predict..."
acc = np.sum(clf.predict(test_feat) == test_label) * 1.0 / np.size(test_label)
print "Accuracy: ", acc

if save_model:
	print "Save model"
	joblib.dump(clf, join(model_save_dir, 'scene.pkl'))

	with open(join(model_save_dir, 'class_names.txt'), 'wb') as f:
		pickle.dump(class_names, f)