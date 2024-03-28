import numpy as np

import torch

from PIL import Image

from PIL import ImageFile

from torch.utils.data import Dataset
import json
from dataloader.ego_utils import video_loader_by_frames

import glob
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob
import cv2

class BaseDatasetFrames(Dataset):

	def __init__(self, unidet_outputs_dir, video_dir, annots_path):

        self.unidet_outputs_dir = unidet_outputs_dir
        self.video_dir = video_dir
        
        with open(annots_path, "r") as f:
             self.annots = json.load(f)['clips']

 	def __len__(self):
        return len(self.annots)

    def __getitem__(self, idx):
        """Returns the 16 frames per segment.
        """

        # step1: Generate Clip-segment name from annottaion.json file content
        seg_annot = self.annots[idx]
        vid_name = seg_annot['clip_uid'] + ".mp4"
        start_frame = seg_annot['action_clip_start_frame']
        end_frame = seg_annot['action_clip_end_frame']
		seg_name = seg_annot['clip_uid'] + "_start_frame_" + str(start_frame) +\
              "_end_frame_" + str(end_frame)

        # step2: Load unidet outputs for the segment, to collect sampled frame id
        with open(self.unidet_outputs_dir + seg_name + ".json", "r") as f:
            unidet_out = json.load(f)        
        frame_ids = list(unidet_out.keys()) # 16 frame ids

        # (num_frames, H, W, C) Convert to torch format.
        img_tensor, im_info, gt_boxes, num_boxes, frame_ids, vid = video_loader_by_frames(self.video_dir, vid_name, frame_ids) 

       
        return img_tensor, im_info, gt_boxes, num_boxes, frame_ids, vid
   

      