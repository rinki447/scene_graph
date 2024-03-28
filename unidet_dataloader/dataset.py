import json
from utils import video_loader_by_frames, prep_im_for_blob, im_list_to_blob
import numpy as np
import torch
from torch.utils.data import Dataset

class SegFileDataset(Dataset):
    def __init__(self, unidet_outputs_dir, video_dir, annots_path):

        self.unidet_outputs_dir = unidet_outputs_dir
        self.video_dir = video_dir
        
        with open(annots_path, "r") as f:
             self.annots = json.load(f)['clips']

    def __len__(self):
        return len(self.annots)
    
    def get_image_scales(self, frames, frame_size=384):
        proc_frames = []
        img_scales = []
        for image in frames:
            img, img_scale = prep_im_for_blob(image, [[[102.9801, 115.9465, 122.7717]]], 
                                              frame_size, frame_size)
            proc_frames.append(img)
            img_scales.append(img_scale)

        return proc_frames, img_scales


    def __getitem__(self, idx):
        """Returns the 16 frames per segment.
        """

        seg_annot = self.annots[idx]

        vid_path = seg_annot['clip_uid'] + ".mp4"
        start_frame = seg_annot['action_clip_start_frame']
        end_frame = seg_annot['action_clip_end_frame']

        seg_name = vid_path.split(".m")[0] + "_start_frame_" + str(start_frame) +\
              "_end_frame_" + str(end_frame)

        # Load unidet outputs for the segment
        with open(self.unidet_outputs_dir + seg_name + ".json", "r") as f:
            unidet_out = json.load(f)        
        
        frame_ids = list(unidet_out.keys()) # 16 frame ids

        # (num_frames, H, W, C) Convert to torch format.
        frames = video_loader_by_frames(self.video_dir, vid_path, frame_ids) 

        # Extract image scales like ActivityNet.
        conv_frames, frame_scales = self.get_image_scales(frames)

        # Everything same as scene-graph code from https://github.com/rinki447/scene_graph/blob/1d60701e1a28da467638a9c517f13cf0f0ac4b36/dataloader/base_dataset.py#L139
        blob = im_list_to_blob(conv_frames)     #(processed_ims)
        im_info = np.array([[blob.shape[1], blob.shape[2], frame_scales[0]]],dtype=np.float32)   # what is im_scales  ?????
        im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)
        img_tensor = torch.from_numpy(blob)
        img_tensor = img_tensor.permute(0, 3, 1, 2)

        gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
        num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)
        #######################################################################################
 

        return img_tensor, im_info, gt_boxes, num_boxes, index, vid_id, path_f, path_list
        # return frames