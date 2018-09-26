# Select qualified images from twitter dataset
import os
import re
import sys
from google_images_download import google_images_download

dest_dir = "../datasets/google"
label_file = open("label_list_256.txt", 'r')
images_per_class = 400

for ANP in label_file:
    segs = re.split("_|\n", ANP)
    ANP_keyword = segs[0] + ' ' + segs[1]
    ANP_dirname = segs[0] + '_' + segs[1]

    print("Downloading images for keyword {}".format(ANP_keyword))
    try:
        response = google_images_download.googleimagesdownload()
        response.download({"keywords": ANP_keyword,
                       "limit": images_per_class,
                       "chromedriver": "/usr/lib/chromium-browser/chromedriver",
                       "format": "jpg",
                       "output_directory": dest_dir,
                       "image_directory": ANP_dirname})
    except:
        print("Failed to download for keyword {}".format(ANP_keyword))
