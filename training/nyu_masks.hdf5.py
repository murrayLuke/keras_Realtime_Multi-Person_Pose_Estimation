#!/usr/bin/env python

from scipy.spatial.distance import cdist
import numpy as np
import cv2
import os
import os.path
import h5py
import json
import scipy.io as sio
import matplotlib.pyplot as plt
from tqdm import tqdm

dataset_dir = os.path.abspath('./dataset/nyu_hand_dataset_v2/dataset/')

tr_anno_path = os.path.join(dataset_dir, "train/joint_data.mat")
tr_img_dir = os.path.join(dataset_dir, "train")

val_anno_path = os.path.join(dataset_dir, "test/joint_data.mat")
val_img_dir = os.path.join(dataset_dir, "test")

print(f"dataset_dir {dataset_dir}\n tr_anno_path {tr_anno_path}\n tr_img_dir {tr_img_dir}\n")
assert(os.path.exists(dataset_dir))
assert(os.path.exists(tr_anno_path))
assert(os.path.exists(tr_img_dir))


datasets = [
    (val_anno_path, val_img_dir, "NYU_Hand_val"),  # it is important to have 'val' in validation dataset name, look for 'val' below
    (tr_anno_path, tr_img_dir, "NYU_Hand")
]


tr_hdf5_path = os.path.join(dataset_dir, "NYU_Hand_train_dataset.h5")
val_hdf5_path = os.path.join(dataset_dir, "NYU_Hand_val_dataset.h5")

def make_mask(image_rec):
    return  np.zeros(image_rec, dtype=np.uint8)

# return none if the image is a bad image otherwise
# return a dictionary which is instance (see code)
def process_image(image_rec, img_id, image_index, img_anns, dataset_type):

    # print("Processing image ID: ", img_id)

    h, w = image_rec

    p = 0
    pers = dict()

    person_center = [img_anns[p]["bbox"][0] + img_anns[p]["bbox"][2] / 2,
                        img_anns[p]["bbox"][1] + img_anns[p]["bbox"][3] / 2]

    pers["objpos"] = person_center
    pers["bbox"] = img_anns[p]["bbox"]
    pers["segment_area"] = img_anns[p]["area"]
    pers["num_keypoints"] = img_anns[p]["num_keypoints"]

    anno = img_anns[p]["keypoints"]

    pers["joint"] = np.zeros((pers["num_keypoints"], 3))
    for part in range( pers["num_keypoints"]):
        pers["joint"][part, 0] = anno[part * 3]
        pers["joint"][part, 1] = anno[part * 3 + 1]

    pers["scale_provided"] = img_anns[p]["bbox"][3] / 368


    if pers["segment_area"] < 32 * 32:
        print(f"skipping {img_id} area {pers['segment_area']} < {32 * 32}")
        return None

    instance = dict()
    instance["dataset"] = dataset_type

    if 'val' in dataset_type:
        isValidation = 1
    else:
        isValidation = 0

    instance["isValidation"] = isValidation
    instance["img_width"] = w
    instance["img_height"] = h
    instance["image_id"] = img_id
    instance["annolist_index"] = image_index # TODO this is wrong for their's
    instance["img_path"] = imgFileName(*img_id)
    instance["objpos"] = [ pers["objpos"] ]
    instance["joints"] = [ pers["joint"].tolist() ]
    instance["scale_provided"] = [ pers["scale_provided"] ]
    return instance


def writeImage(grp, img_grp, data, img, mask_miss, count, image_id, mask_grp=None):

    serializable_meta = data
    serializable_meta['count'] = count

    img_key = imgFileName(*image_id)
    if not img_key in img_grp:

        if mask_grp is None:
            img_and_mask = np.concatenate((img, mask_miss[..., None]), axis=2)
            img_ds = img_grp.create_dataset(img_key, data=img_and_mask, chunks=None)
        else:
            _, img_bin = cv2.imencode(".jpg", img)
            _, img_mask = cv2.imencode(".png", mask_miss)
            img_ds1 = img_grp.create_dataset(img_key, data=img_bin, chunks=None)
            img_ds2 = mask_grp.create_dataset(img_key, data=img_mask, chunks=None)


    key = '%07d' % count
    required = { 'image':img_key, 'joints': serializable_meta['joints'], 'objpos': serializable_meta['objpos'], 'scale_provided': serializable_meta['scale_provided'] }
    ds = grp.create_dataset(key, data=json.dumps(required), chunks=None)
    ds.attrs['meta'] = json.dumps(serializable_meta)

    # print('Writing sample %d' % count)


def process():

    tr_h5 = h5py.File(tr_hdf5_path, 'w')
    tr_grp = tr_h5.create_group("dataset")
    tr_write_count = 0
    tr_grp_img = tr_h5.create_group("images")
    tr_grp_mask = tr_h5.create_group("masks")

    val_h5 = h5py.File(val_hdf5_path, 'w')
    val_grp = val_h5.create_group("dataset")
    val_write_count = 0
    val_grp_img = val_h5.create_group("images")
    val_grp_mask = val_h5.create_group("masks")

    for idx, ds in enumerate(datasets):

        # absolute path to annotations
        anno_path = ds[0]

        # absolute path to directory containing images
        img_dir = ds[1]

        # the dataset type
        dataset_type = ds[2]  
        # the actual dataset as a dictionary
        dataset = sio.loadmat(anno_path)
        for image_index, kinect_idx in tqdm(generateTrainIds(dataset_type)):
            # image annotations TODO define what these are
            # bbox
            # area
            # num_keypoints
            # keypoints

            img_anns = getAnnotation(dataset, image_index, kinect_idx)

            # image filename in the image directory
            img_filename = imgFileName(image_index, kinect_idx)

            # numpy array of the image
            img = cv2.imread(os.path.join(img_dir, img_filename))

            # width and height of the image (w, h)
            image_rec = np.shape(img)[:-1]

            # image id (imgidx, kinectidx)
            img_id = (image_index, kinect_idx)

            data = process_image(image_rec, img_id, image_index, img_anns, dataset_type)
            if data is None:
                continue

            mask_miss = make_mask(image_rec) 

            if data['isValidation']:
                writeImage(val_grp, val_grp_img, data, img, mask_miss, val_write_count, img_id, val_grp_mask)
                val_write_count += 1
            else:
                writeImage(tr_grp, tr_grp_img, data, img, mask_miss, tr_write_count, img_id, tr_grp_mask)
                tr_write_count += 1


    tr_h5.close()
    val_h5.close()

def generateTrainIds(dataset_type):
    # train is 3 x 72757 = 218271
    # val is 3 x 8252 = 24756

    if 'val' in dataset_type:
        imgMax = 3
        kinectMax = 1000
    else:
        imgMax = 3
        kinectMax = 5000    

    ids = []
    for imgIdx in range(1, imgMax + 1):
        for knctIdx in range(1, kinectMax + 1):
            ids.append((imgIdx, knctIdx))

    return ids

def getAnnotation(dataset, imgIdx, kinectIdx):
    annoDic = {}
    joint_uvd = dataset['joint_uvd']
    jnt_uvd = np.squeeze(joint_uvd[imgIdx - 1, kinectIdx - 1, :, :]) # joint u, v, depth data
    annoDic["num_keypoints"] = np.size(jnt_uvd,0) 
    x = jnt_uvd[:, 0]
    y = jnt_uvd[:, 1]
    x0,x1,y0,y1 = np.min(x), np.max(x), np.min(y), np.max(y)
    annoDic['area'] = (x1-x0)*(y1-y0)
    annoDic['bbox'] = [x0,y0,x1-x0,y1-y0]
    # visible/invisible
    # COCO - Each keypoint has a 0-indexed location x,y and a visibility flag v defined as v=0: not labeled (in which case x=y=0), v=1: labeled but not visible, and v=2: labeled and visible.
    # OURS - # 3 never marked up in this dataset, 2 - not marked up in this person, 1 - marked and visible, 0 - marked but invisible
    jnt_uvd[:, 2] = 1
    annoDic['keypoints'] = jnt_uvd.flatten()
    
    return [annoDic]

def imgFileName(imgIdx: int, kinectIdx: int) -> str:
    return 'rgb_{k}_{f:07d}.png'.format(k = imgIdx, f=kinectIdx)


if __name__ == '__main__':
    process()
