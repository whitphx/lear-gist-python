import sys
from os import listdir, mkdir
from os.path import exists, isdir, join
import numpy as np
from skimage import transform
from PIL import Image
import gist

argv = sys.argv
argc = len(argv)

if (argc < 3):
	print "Not enough arguments."
	quit(1)


input_dir = argv[1]
output_dir = argv[2]

if (not exists(output_dir)):
	mkdir(output_dir)

imsize = (128, 128)

features = {}

filenames = listdir(input_dir)
for filename in filenames:
	filepath = join(input_dir, filename)

	try:
		pilimg = Image.open(filepath)
	except:
		continue

	img = np.asarray(pilimg)
	img_resized = transform.resize(img, imsize, preserve_range=True).astype(np.uint8)
	desc = gist.extract(img_resized)

	class_name = filename.split('_')[0]

	if (class_name in features):
		features[class_name] = np.vstack((features[class_name], desc))
	else:
		features[class_name] = np.atleast_2d(desc)

for class_name, desc_mat in features.items():
	np.save(join(output_dir, class_name+'.npy'), desc_mat)