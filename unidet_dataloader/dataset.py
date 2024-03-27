import json
from utils import video_loader_by_frames
from torch.utils.data import Dataset

class SegFileDataset(Dataset):
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

        return frames