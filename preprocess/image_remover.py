import os
import sys
import cPickle as pickle
from PIL import Image

dirname = "../datasets/sentibank_flickr/image/"
image_cnt = 0
error_cnt = 0

for subdir, dirs, files in os.walk(dirname):
    for f in files:  
        if image_cnt % 1000 == 0: 
            sys.stdout.write('.') 
            sys.stdout.flush()

        fn = os.path.join(subdir, f)
        if fn.find("txt") != -1:
            continue;

        try:
            im = Image.open(fn)
        except:
            os.remove(fn)
            error_cnt += 1 
            continue;

        image_cnt += 1

print ''

print "number of images and errorcounts:", image_cnt, error_cnt

