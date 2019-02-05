
"""

Edited for multiclass boxes

Takes a JSON file and visualizes the annotation boxes on images.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys
sys.path.append('./tools')
import numpy as np
import os, cv2
import argparse
import os.path as osp
import time
import skvideo.io
import json
import csv
from six.moves import xrange
from PIL import Image
from tqdm import tqdm

from matplotlib import pyplot as plt


JSON_FILE = '' #'data/CS6_annot/cs6-subset-gt_face_train_annot_coco_style.json'
# OUT_DIR = '/mnt/nfs/work1/elm/arunirc/Data/CS6_annots'
OUT_DIR = '' #'Outputs/visualizations/'

DEBUG = False

def parse_args():
    parser = argparse.ArgumentParser(description='Creating CS6 ground truth data')
    parser.add_argument(
        '--output_dir', help='directory for saving outputs',
        default=OUT_DIR, type=str
    )
    parser.add_argument(
        '--json_file', help='Name of JSON file', default=JSON_FILE
    )
    parser.add_argument(
        '--imdir', help="root directory for loading dataset images",
        default='data/CS6_annot', type=str)
    return parser.parse_args()




#_GREEN = (18, 127, 15)
#color_dict = {'red': (0,0,225), 'green': (0,255,0), 'yellow': (0,255,255), 
#              'blue': (255,0,0), '_GREEN':(18, 127, 15), '_GRAY': (218, 227, 218), 'COL1': (255,255,0),'COL2':(255,0,255)}

color_dict = {i:(np.random.randint(255),np.random.randint(255),np.random.randint(255)) for i in range(500)}

# -----------------------------------------------------------------------------------
def draw_detection_list(im, dets, labels, sources=None):
# -----------------------------------------------------------------------------------
    """ Draw bounding boxes on a copy of image and return it.
        [x0 y0 w h conf_score]
    """
    im_det = im.copy()
    if dets.size == 0:
        return im_det
    if dets.ndim == 1:
        dets = dets[np.newaxis,:] # handle single detection case

    # format into [xmin, ymin, xmax, ymax]
    dets[:, 2] = dets[:, 2] + dets[:, 0]
    dets[:, 3] = dets[:, 3] + dets[:, 1]

    for i, det in enumerate(dets):
        bbox = dets[i, :4]
        x0, y0, x1, y1 = [int(x) for x in bbox]
        
        # TEMP: if 'source' is available, set yellow for HP and green for dets -- assuming 1 class
        if not (sources is None):
            if sources[i] == 1:
                col = 'green'
            elif sources[i] == 2:
                col = 'yellow'
        else:
            #col = list(sorted(color_dict.keys()))[labels[i]] #-- standard multi class, not considering the source
            col = labels[i]
        line_color = color_dict[col]
        cv2.rectangle(im_det, (x0, y0), (x1, y1), line_color, thickness=2)
        
    return im_det

if __name__ == '__main__':
    

    args = parse_args()

    with open(args.json_file) as f:
        ann_dict = json.load(f)
    print(ann_dict.keys())
    #input(ann_dict['images'][0])
    #input(ann_dict['annotations'][0])

    #out_dir = osp.join(args.output_dir, 
    #                   osp.splitext(osp.basename(args.json_file))[0])
    out_dir = args.output_dir

    if not osp.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    
    i = 0
    np.random.seed(0)
    imlist = ann_dict['images']
    np.random.shuffle(imlist)
    for img_annot in tqdm(imlist):#ann_dict['images']):
        image_name = img_annot['file_name']
        image_id = img_annot['id']
        bboxes = [x['bbox'] for x in ann_dict['annotations'] if x['image_id'] == image_id]
        labels = [x['category_id'] for x in ann_dict['annotations'] if x['image_id'] == image_id]
        
        # get source -- gt, dets or hp
        sources = None
        if 'source' in ann_dict['annotations'][0].keys():
            sources = [x['source'] for x in ann_dict['annotations'] if x['image_id'] == image_id]
            #input('-->'+str(sources[0]))
       
        #print('>>>',osp.join(args.imdir, image_name))
        im = cv2.imread(osp.join(args.imdir, image_name))
        if im is None:
            print('>>>',osp.join(args.imdir, image_name))
            continue
        assert im.size > 0
        im_det = draw_detection_list(im, np.array(bboxes), labels, sources=sources)
        
        #if im_det is None:
        #    print('>>>',image_name)
        #    plt.imshow(im)
        #    plt.show()

        out_path = osp.join(out_dir, image_name.replace('/', '_'))
        cv2.imwrite(out_path, im_det)
        i += 1
        if i == 5000:
            break
