import torch.utils.data as data
import contextlib
import io
import logging
import os
import pycocotools.mask as mask_util
import os.path as osp
import cv2
import h5py
import numpy as np
from .config import cfg
import random
import torch





def get_label_map():

    return {x+1: x+1 for x in range(3)}

logger = logging.getLogger(__name__)

class COCOAnnotationTransform(object):
    """Transforms a COCO annotation into a Tensor of bbox coords and label index
    Initilized with a dictionary lookup of classnames to indexes
    """
    def __init__(self):
        self.label_map = get_label_map()

    def __call__(self, target, width, height):
        """
        Args:
            target (dict): COCO target json annotation as a python dict
            height (int): height
            width (int): width
        Returns:
            a list containing lists of bounding boxes  [bbox coords, class idx]
        """
        scale = np.array([width, height, width, height])
        res = []
        for obj in target:
            if 'bbox' in obj:
                bbox = obj['bbox']
                label_idx = obj['category_id']
                origin=obj['motion']['origin']
                axis=obj['motion']['axis']
                if label_idx >= 0:
                    label_idx = self.label_map[label_idx] - 1
                final_box = list(np.array([bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]])/scale)
                final_box.append(label_idx)
                final_box.append(origin[0])
                final_box.append(origin[1])
                final_box.append(origin[2])
                final_box.append(axis[0])
                final_box.append(axis[1])
                final_box.append(axis[2])
                res += [final_box]  # [xmin, ymin, xmax, ymax, label_idx]
            else:
                print("No bbox found for object ", obj)

        return res

# Adapted detectron2.data.datasets.coco.load_coco_json to add custom img annotations
def load_motion_json(json_file, image_root, dataset_name=None, extra_annotation_keys=None):
	"""
	Load a json file with COCO instances annotation format + motionnet hacks

	Args:
		json_file (str): full path to the json file in COCO instances annotation format.
		image_root (str or path-like): the directory where the images in this json file exists.
		dataset_name (str): the name of the dataset (e.g., coco_2017_train).

	Returns:
		list[dict]: a list of dicts in Detectron2 standard dataset dicts format. (See
		`Using Custom Datasets </tutorials/datasets.html>`_ )

	Notes:
		1. This function does not read the image files.
		   The results do not have the "image" field.
	"""
	from pycocotools.coco import COCO

	with contextlib.redirect_stdout(io.StringIO()):
		coco_api = COCO(json_file)

	id_map = None
	if dataset_name is not None:
		meta = MetadataCatalog.get(dataset_name)
		cat_ids = sorted(coco_api.getCatIds())
		cats = coco_api.loadCats(cat_ids)
		# The categories in a custom json file may not be sorted.
		thing_classes = [c["name"] for c in sorted(cats, key=lambda x: x["id"])]
		meta.thing_classes = thing_classes

		id_map = {v: i for i, v in enumerate(cat_ids)}
		meta.thing_dataset_id_to_contiguous_id = id_map

	# sort indices for reproducible results
	img_ids = sorted(coco_api.imgs.keys())
	imgs = coco_api.loadImgs(img_ids)
	anns = [coco_api.imgToAnns[img_id] for img_id in img_ids]
	imgs_anns = list(zip(imgs, anns))

	logger.info(f"Loaded {len(imgs_anns)} images in COCO format from {json_file}")

	dataset_dicts = []

	ann_keys = ["iscrowd", "bbox", "keypoints", "category_id", "motion", "object_key"] + (extra_annotation_keys or [])

	for (img_dict, anno_dict_list) in imgs_anns:
		record = {}
		record["file_name"] = os.path.join(image_root, img_dict["file_name"])
		record["height"] = img_dict["height"]
		record["width"] = img_dict["width"]
		image_id = record["image_id"] = img_dict["id"]
		# 2DMotion changes: add extra annotations from img_dict
		# Ideally, this would be handled with something like "extra_img_keys"
		record["camera"] = img_dict["camera"]
		record["depth_file_name"] = img_dict["depth_file_name"]
		# record["label"] = img_dict["label"]

		objs = []
		for anno in anno_dict_list:
			assert anno["image_id"] == image_id

			assert anno.get("ignore", 0) == 0, '"ignore" in COCO json file is not supported.'

			obj = {key: anno[key] for key in ann_keys if key in anno}

			segm = anno.get("segmentation", None)
			if segm:  # either list[list[float]] or dict(RLE)
				if isinstance(segm, dict):
					if isinstance(segm["counts"], list):
						# convert to compressed RLE
						segm = mask_util.frPyObjects(segm, *segm["size"])
				else:  # filter out invalid polygons (< 3 points)
					segm = [p for p in segm if len(p) % 2 == 0 and len(p) >= 6]
					if len(segm) == 0:
						continue  # ignore this instance
				obj["segmentation"] = segm
			
			keypts = anno.get("keypoints", None)
			if keypts:  # list[int]
				for idx, v in enumerate(keypts):
					if idx % 3 != 2:
						# COCO's segmentation coordinates are floating points in [0, H or W],
						# but keypoint coordinates are integers in [0, H-1 or W-1]
						# Therefore we assume the coordinates are "pixel indices" and
						# add 0.5 to convert to floating point coordinates.
						keypts[idx] = v + 0.5
				obj["keypoints"] = keypts

			# obj["bbox_mode"] = BoxMode.XYWH_ABS
			if id_map:
				obj["category_id"] = id_map[obj["category_id"]]
			objs.append(obj)
		record["annotations"] = objs
		dataset_dicts.append(record)

	return dataset_dicts








class OPDmultiDetection(data.Dataset):


    def __init__(self, image_path,info_file, transform=None,
                 target_transform=None,
                 dataset_name='OPDmulti', has_gt=True, mode='train'):
        
        from pycocotools.coco import COCO
        
        self.coco=COCO(info_file)
		
        self.ids = list(self.coco.imgToAnns.keys())
        if len(self.ids) == 0 or not has_gt:
            self.ids = list(self.coco.imgs.keys())
        
        
        self.root = image_path
        
        self.transform = transform
        self.target_transform = COCOAnnotationTransform()

        self.name = dataset_name

        self.has_gt = has_gt
        self.filenames_map = {}
        self.load_h5(image_path, mode)
		

    def load_h5(self, base_dir, dir):

        h5file = h5py.File(f'{base_dir}/{dir}.h5')
        self.images = h5file[f'{dir}_images']
        self.filenames = h5file[f'{dir}_filenames']
        num_images = self.filenames.shape[0]
        for i in range(num_images):
            self.filenames_map[self.filenames[i].decode('utf-8')] = i

    def __getitem__(self, index):

        im, gt, masks, h, w, num_crowds = self.pull_item(index)
        return im, (gt, masks, num_crowds)

    def __len__(self):
        return len(self.ids)
    

    def pull_item(self, index):
		
        img_id = self.ids[index]

        if self.has_gt:
            ann_ids = self.coco.getAnnIds(imgIds=img_id)
            target = [x for x in self.coco.loadAnns(ann_ids) if x['image_id'] == img_id]
        else:
            target = []
        
        crowd  = [x for x in target if     ('iscrowd' in x and x['iscrowd'])]
        target = [x for x in target if not ('iscrowd' in x and x['iscrowd'])]
        num_crowds = len(crowd)

        file_name = self.coco.loadImgs(img_id)[0]['file_name']
		
        if file_name.startswith('COCO'):
            file_name = file_name.split('_')[-1]
        
        img = self.images[self.filenames_map[file_name]].astype(np.float32)
		
        height, width, _ = img.shape
		
        if len(target) > 0:
            # Pool all the masks for this image into one [num_objects,height,width] matrix
            masks = [self.coco.annToMask(obj).reshape(-1) for obj in target]
            masks = np.vstack(masks)
            masks = masks.reshape(-1, height, width)
			
        if self.target_transform is not None and len(target) > 0:
            target = self.target_transform(target, width, height)


        target = np.array(target)
        if self.transform is not None:
            if len(target) > 0:
                target = np.array(target)
                img, masks, boxes, labels = self.transform(img, masks, target[:, :4],
                    {'num_crowds': num_crowds, 'labels': target[:, 4]})
            
                # I stored num_crowds in labels so I didn't have to modify the entirety of augmentations
                num_crowds = labels['num_crowds']
                labels     = labels['labels']
                
                # print(boxes.shape,labels.shape,target.shape)
                target = np.hstack((boxes, np.expand_dims(labels, axis=1),target[:, 5:]))
            else:
                img, _, _, _ = self.transform(img, np.zeros((1, height, width), dtype=np.float), np.array([[0, 0, 1, 1]]),
                    {'num_crowds': 0, 'labels': np.array([0])})
                masks = None
                target = None			

        if target.shape[0] == 0:
            print('Warning: Augmentation output an example with no ground truth. Resampling...')
            return self.pull_item(random.randint(0, len(self.ids)-1))
		

        return torch.from_numpy(img).permute(2, 0, 1), target, masks, height, width, num_crowds 
