from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
try:
  import cPickle as pickle
except ImportError:
  import pickle
import cv2
import numpy as np
from datasets.imdb import imdb
from model.utils.config import cfg
import scipy.sparse
import uuid
from pyvgtools.ade import VG
from pyvgtools.adeeval import VGeval
try:
  xrange          # Python 2
except NameError:
  xrange = range  # Python 3

import json
import argparse
from collections import OrderedDict

class ade(imdb):
  def __init__(self, image_set, count=5):
    imdb.__init__(self, 'ade_%s_%d' % (image_set, count))
    self.config = {'use_salt': True,
                   'cleanup': True}
    self._image_set = image_set
    self._root_path = osp.join(cfg.DATA_DIR, 'ADE')
    self._name_file = osp.join(self._root_path, 'sharename.txt')
    self._anno_file = osp.join(self._root_path, self._image_set + '.txt')
    with open(self._anno_file) as fid:
      image_index = fid.readlines()
      self._image_index = [ii.strip() for ii in image_index]
    self._raw_counts = []
    with open(self._name_file) as fid:
      raw_names_count = fid.readlines()
      self._raw_names = [n.split(',')[0].strip() for n in raw_names_count]
      self._len_raw = len(self._raw_names)
      self._raw_counts = np.array([int(n.split(',')[1].strip()) for n in raw_names_count])

    # First class is always background
    self._ade_inds = [0] + list(np.where(self._raw_counts >= count)[0])
    self._classes = ['__background__']

    for idx in self._ade_inds:
      if idx == 0:
        continue
      ade_name = self._raw_names[idx]
      self._classes.append(ade_name)

    self._classes = tuple(self._classes)
    self._class_to_ind = dict(zip(self.classes, list(range(self.num_classes))))

    self.set_proposal_method('gt')
    self.competition_mode(False)

  def _load_text(self, text_path):
    class_keys = {}
    with open(text_path) as fid:
      lines = fid.readlines()
      for line in lines:
        columns = line.split('#')
        key = '%s_%s' % (columns[0].strip(), columns[1].strip())
        # Just get the class ID
        # class_name = columns[4].strip().replace(' ', '_')
        ####################### merge class ###########################
        class_name = columns[4].strip().replace(' ', '_')
        if class_name.find('_court') >= 0:
          class_name = 'court'
        if class_name == 'short_pants':
          class_name = 'pants'
        if len(class_name) > 3:
          if class_name[-2:] == 'es' and (
                  class_name[-3] in ['s', 'x'] or (class_name[-4] + class_name[-3]) in ['sh', 'ch']):
            class_name = class_name[:-2]
          elif class_name[-3:] == 'ves' and class_name not in ['waves', 'eaves']:
            if class_name == 'knives':
              class_name = 'knife'
            else:
              class_name = class_name[:-3] + 'f'
          elif class_name[-1] == 's' and class_name[-2] != 's' and class_name not in [
            'scissors', 'pants', 'sunglasses', 'chest_of_drawers', 'goggles', 'tennis']:
            class_name = class_name[:-1]
        elif class_name[-1] == 's' and class_name[-2] != 's' and class_name not in [
          'scissors', 'pants', 'sunglasses', 'chest_of_drawers', 'goggles', 'tennis']:
          class_name = class_name[:-1]
        ############################## merge done ############################

        if class_name in self._class_to_ind:
          class_keys[key] = self._class_to_ind[class_name]
      total_num_ins = len(lines)

    return class_keys, total_num_ins

  def _load_annotation(self):
    gt_roidb = []

    for i in xrange(self.num_images):
      image_path = self.image_path_at(i)
      if i % 10 == 0:
        print(image_path)
      # Estimate the number of objects from text file
      text_path = image_path.replace('.jpg', '_atr.txt')
      class_keys, total_num_ins = self._load_text(text_path)

      valid_num_ins = 0
      boxes = np.zeros((total_num_ins, 4), dtype=np.uint16)
      gt_classes = np.zeros((total_num_ins), dtype=np.int32)
      overlaps = np.zeros((total_num_ins, self.num_classes), dtype=np.float32)
      seg_areas = np.zeros((total_num_ins), dtype=np.float32)

      # First, whole objects
      label_path = image_path.replace('.jpg', '_seg.png')
      seg = cv2.imread(label_path)
      height, width, _ = seg.shape

      # OpenCV has reversed RGB
      instances = seg[:, :, 0]
      unique_ins = np.unique(instances)

      for t, ins in enumerate(list(unique_ins)):
        if ins == 0:
          continue
        key = '%03d_%d' % (t, 0)
        if key in class_keys:
          ins_seg = np.where(instances == ins)
          x1 = ins_seg[1].min()
          x2 = ins_seg[1].max()
          y1 = ins_seg[0].min()
          y2 = ins_seg[0].max()
          boxes[valid_num_ins, :] = [x1, y1, x2, y2]
          gt_classes[valid_num_ins] = class_keys[key]
          seg_areas[valid_num_ins] = ins_seg[0].shape[0]
          valid_num_ins += 1

      # Then deal with parts
      level = 1
      while True:
        part_path = image_path.replace('.jpg', '_parts_%d.png' % level)
        if osp.exists(part_path):
          seg = cv2.imread(part_path)
          instances = seg[:, :, 0]
          unique_ins = np.unique(instances)

          for t, ins in enumerate(list(unique_ins)):
            if ins == 0:
              continue
            key = '%03d_%d' % (t, level)
            if key in class_keys:
              ins_seg = np.where(instances == ins)
              x1 = ins_seg[1].min()
              x2 = ins_seg[1].max()
              y1 = ins_seg[0].min()
              y2 = ins_seg[0].max()
              boxes[valid_num_ins, :] = [x1, y1, x2, y2]
              gt_classes[valid_num_ins] = class_keys[key]
              seg_areas[valid_num_ins] = ins_seg[0].shape[0]
              valid_num_ins += 1

          level += 1
        else:
          break

      boxes = boxes[:valid_num_ins, :]
      gt_classes = gt_classes[:valid_num_ins]
      seg_areas = seg_areas[:valid_num_ins]
      overlaps = overlaps[:valid_num_ins, :]
      for ix in range(valid_num_ins):
        overlaps[ix, gt_classes[ix]] = 1.0
      overlaps = scipy.sparse.csr_matrix(overlaps)

      gt_roidb.append({'width': width,
                      'height': height,
                      'boxes' : boxes,
                      'gt_classes': gt_classes,
                      'gt_overlaps': overlaps,
                      'flipped' : False,
                      'seg_areas': seg_areas})
    return gt_roidb

  def image_id_at(self, i):
    return self._image_index[i]

  def image_path_at(self, i):
    return osp.join(self._root_path, self._image_index[i])

  def image_path_from_index(self, index):
    """
    Construct an image path from the image's "index" identifier.
    """
    image_path = os.path.join(self._root_path, index)
    assert os.path.exists(image_path), \
      'Path does not exist: {}'.format(image_path)
    return image_path

  def gt_roidb(self):
    cache_file = osp.join(self.cache_path, self.name + '_gt_roidb.pkl')
    image_file = osp.join(self.cache_path, self.name + '_gt_image.pkl')
    if osp.exists(cache_file) and osp.exists(image_file):
      with open(cache_file, 'rb') as fid:
        gt_roidb = pickle.load(fid)
      print('{} gt roidb loaded from {}'.format(self.name, cache_file))
      with open(image_file, 'rb') as fid:
        self._image_index = pickle.load(fid)
      print('{} gt image loaded from {}'.format(self.name, image_file))
      return gt_roidb

    gt_roidb = self._load_annotation()
    with open(cache_file, 'wb') as fid:
      pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt roidb to {}'.format(cache_file))
    with open(image_file, 'wb') as fid:
      pickle.dump(self._image_index, fid, pickle.HIGHEST_PROTOCOL)
    print('wrote gt image to {}'.format(image_file))
    return gt_roidb

  # Do some left-right flipping here
  def _find_flipped_classes(self):
    self._flipped_classes = np.arange(self.num_classes, dtype=np.int32)
    for i, cls_name in enumerate(self.classes):
      if cls_name.startswith('left_'):
        query = cls_name.replace('left_', 'right_')
        idx = self._class_to_ind[query]
        # Swap for both left and right
        self._flipped_classes[idx] = i
        self._flipped_classes[i] = idx

  def append_flipped_images(self):
    self._find_flipped_classes()
    num_images = self.num_images
    widths = self._get_widths()
    for i in range(num_images):
      boxes = self.roidb[i]['boxes'].copy()
      oldx1 = boxes[:, 0].copy()
      oldx2 = boxes[:, 2].copy()
      boxes[:, 0] = widths[i] - oldx2 - 1
      boxes[:, 2] = widths[i] - oldx1 - 1
      assert (boxes[:, 2] >= boxes[:, 0]).all()
      entry = {'width': widths[i],
               'height': self.roidb[i]['height'],
               'boxes': boxes,
               'gt_classes': self._flipped_classes[self.roidb[i]['gt_classes']],
               'gt_overlaps': self.roidb[i]['gt_overlaps'],
               'flipped': True,
               'seg_areas': self.roidb[i]['seg_areas']}
      self.roidb.append(entry)
    self._image_index = self._image_index * 2

  def evaluate_detections(self, all_boxes, output_dir, is_savegt=True):
    res_file = osp.join(output_dir, ('detections_' +
                                     self._image_set +
                                     '_results'))
    if self.config['use_salt']:
      res_file += '_{}'.format(str(uuid.uuid4()))
    res_file += '.json'
    self._write_vg_results_file(all_boxes, res_file)
    # save gt to json
    res_file_gt = osp.join(self._root_path, 'val_gt.json')
    if is_savegt:
      self._write_gt_results_file(res_file_gt)
    # Only do evaluation on non-test sets
    if self._image_set.find('test') == -1:
      self._do_detection_eval(res_file_gt.split('/')[-1], res_file.split('/')[-1], output_dir)
    # Optionally cleanup results json file
    if self.config['cleanup']:
      os.remove(res_file)

  def _print_detection_eval_metrics(self, vg_eval):
    IoU_lo_thresh = 0.5
    IoU_hi_thresh = 0.95

    def _get_thr_ind(vg_eval, thr):
      ind = np.where((vg_eval.params.iouThrs > thr - 1e-5) &
                     (vg_eval.params.iouThrs < thr + 1e-5))[0][0]
      iou_thr = vg_eval.params.iouThrs[ind]
      assert np.isclose(iou_thr, thr)
      return ind

    ind_lo = _get_thr_ind(vg_eval, IoU_lo_thresh)
    ind_hi = _get_thr_ind(vg_eval, IoU_hi_thresh)
    # precision has dims (iou, recall, cls, area range, max dets)
    # area range index 0: all area ranges
    # max dets index 2: 100 per image
    precision = \
      vg_eval.eval['precision'][ind_lo:(ind_hi + 1), :, :, 0, 2]
    ap_default = np.mean(precision[precision > -1])
    print(('~~~~ Mean and per-category AP @ IoU=[{:.2f},{:.2f}] '
           '~~~~').format(IoU_lo_thresh, IoU_hi_thresh))
    print('{:.1f}'.format(100 * ap_default))
    for cls_ind, cls in enumerate(self.classes):
      if cls == '__background__':
        continue
      # minus 1 because of __background__
      precision = vg_eval.eval['precision'][ind_lo:(ind_hi + 1), :, cls_ind - 1, 0, 2]
      ap = np.mean(precision[precision > -1])
      print('{}: {:.1f}'.format(cls, 100 * ap))

    print('~~~~ Summary metrics ~~~~')
    vg_eval.summarize()

  def _do_detection_eval(self, res_file_gt, res_file, output_dir):
    vg_gt = VG(self._root_path, res_file_gt)
    vg_dt = VG(output_dir, res_file)
    vg_eval = VGeval(vg_gt, vg_dt, self._class_to_ind)
    vg_eval.evaluate()
    vg_eval.accumulate()
    self._print_detection_eval_metrics(vg_eval)
    # eval_file = osp.join(output_dir, 'detection_results.pkl')
    # with open(eval_file, 'wb') as fid:
    #     pickle.dump(vg_eval, fid, pickle.HIGHEST_PROTOCOL)
    # print('Wrote VG eval results to: {}'.format(eval_file))

  def _vg_results_one_image(self, boxes, index, cnt):
    results = []
    for cls_ind, cls in enumerate(self.classes[1:]):
      dets = boxes[cls_ind].astype(np.float)
      if dets == []:
        continue
      scores = dets[:, -1]
      xs = dets[:, 0]
      ys = dets[:, 1]
      ws = dets[:, 2] - xs + 1
      hs = dets[:, 3] - ys + 1
      results.extend(
        [{'image_id': index,
          'category_id': self._class_to_ind[cls],
          'x': xs[k],
          'y': ys[k],
          'w': ws[k],
          'h': hs[k],
          'object_id': cnt + k,
          'synsets': [cls],
          'score': scores[k]} for k in range(dets.shape[0])])
      cnt += dets.shape[0]
    return results, cnt

  def _write_vg_results_file(self, all_boxes, res_file):
    results = []
    cnt = 0
    for img_ind, index in enumerate(self.image_index):
      print('Collecting {} results ({:d}/{:d})'.format(index, img_ind + 1,
                                                       len(self.image_index)))
      image = {'image_id': index}
      objects, cnt = self._vg_results_one_image([all_boxes[cls_ind][img_ind]
                                                 for cls_ind in range(1, len(self.classes))],
                                                index, cnt)
      image['objects'] = objects
      results.append(image)
    print('Writing results json to {}'.format(res_file))
    with open(res_file, 'w') as fid:
      json.dump(results, fid)

  def _write_gt_results_file(self, res_file):
    results = []
    cnt = 0
    for img_ind, index in enumerate(self.image_index):
      print('Collecting {} results ({:d}/{:d})'.format(index, img_ind + 1,
                                                       len(self.image_index)))
      image = {'image_id': index}
      objects = []
      roi = self.roidb[img_ind]
      xs = roi['boxes'][:, 0]
      ys = roi['boxes'][:, 1]
      ws = roi['boxes'][:, 2] - xs + 1
      hs = roi['boxes'][:, 3] - ys + 1
      class_ = roi['gt_classes']
      objects.extend(
        [{'image_id': index,
          'category_id': int(class_[k]),
          'x': float(xs[k]),
          'y': float(ys[k]),
          'w': float(ws[k]),
          'h': float(hs[k]),
          'object_id': cnt + k,
          'synsets': [self.classes[class_[k]]]
          } for k in range(len(class_))])
      cnt += len(roi['boxes'])

      image['objects'] = objects
      # image['width'] = roi['width']
      # image['height'] = roi['height']
      results.append(image)
    print('Writing results json to {}'.format(res_file))
    with open(res_file, 'w') as fid:
      json.dump(results, fid)

  def competition_mode(self, on):
    if on:
      self.config['use_salt'] = False
      self.config['cleanup'] = False
    else:
      self.config['use_salt'] = True
      self.config['cleanup'] = True



parser = argparse.ArgumentParser(description='Convert datasets to COCO-style JSON format')
parser.add_argument('--skip_flipped', default=False, action='store_true',
                    help='set to true to keep the flipped (augmented) the images')
parser.add_argument('--output_dir', default='', type=str,
                    help='directory to save the generated json files')

parser.set_defaults(bottleneck=True)
parser.set_defaults(augment=True)




if __name__ == '__main__':
   
    global args, best_prec1, exp_name
    args = parser.parse_args()
    print(args)
    
    dataset_name = 'ade'
    skip_flipped = args.skip_flipped

    for image_set in ['train','val']:
        ade_obj = ade(image_set)
        annot_list = ade_obj._load_annotation()
        images = []
        annotations = []
        ann_id = 0
        for i in xrange(ade_obj.num_images):
            flipped = annot_list[i]['flipped']
            if flipped and skip_flipped:
                continue
            # add to images
            image_path = ade_obj.image_path_at(i)
            image_annots = annot_list[i]
            image = {}
            image['id'] = i
            image['width'] = int(image_annots['width'])
            image['height'] = int(image_annots['height'])
            image['file_name'] = image_path
            image = OrderedDict(image)
            images.append(image)
            # add to annotations
            for bbox_ann,category_id in zip(annot_list[i]['boxes'],annot_list[i]['gt_classes']):
                ann = {}
                ann['id'] = ann_id
                ann['image_id'] = image['id']
                ann['segmentation'] = []
                ann['category_id'] = int(category_id)
                ann['iscrowd'] = 0
                bbox = list(map(int,bbox_ann))
                ann['bbox'] = [bbox[0],bbox[1],bbox[2]-bbox[0],bbox[3]-bbox[1]]
                ann['area'] = int(bbox[2]-bbox[0])*int(bbox[3]-bbox[1])
                ann['dataset'] = dataset_name
                ann = OrderedDict(ann)
                annotations.append(ann)
                ann_id += 1

        print('Num classes:',ade_obj.num_classes)
        print('Num images:',ade_obj.num_images)
        save_file = dataset_name+'_'+image_set
        if args.skip_flipped:
            save_file += '_with_flips'
        save_file += '.json'
        save_file = os.path.join(args.output_dir,save_file)
        json_dataset = {}
        json_dataset['images'] = images
        json_dataset['annotations'] = annotations
        json_dataset['categories'] = ade_obj.classes
        json_dataset = OrderedDict(json_dataset)
        with open(save_file,'w') as f:
            json.dump(json_dataset,f)


