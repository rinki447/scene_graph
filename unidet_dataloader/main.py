"""
#!/bin/bash -l

EGO4D_CLIPS_DIR="/home/csres/athatipelli/anirudh/ego4d_data/v2/clips/"
UNIDET_DIR="/home/csres/athatipelli/anirudh/unidet_outputs/"
EGO4D_ANNOTS_PATH="/data/AmitRoyChowdhury/ego4d_data/v2/annotations/fho_lta_train.json"

# cd /home/csres/athatipelli/anirudh/cluster_backup/UniDet/
python3 extract_ego4d_segment_unidet_outputs.py --video-dir ${EGO4D_CLIPS_DIR} \
    --unider_outputs_dir ${UNIDET_DIR} --ego4d_annots_path ${EGO4D_ANNOTS_PATH} \

"""

import argparse
import os, json
import tqdm

from dataset import SegFileDataset


def main(args):

    ego4d_annots_path = args.ego4d_annots_path
    unidet_outputs_dir = args.unidet_outputs_dir
    video_dir = args.video_dir

    num_frames = args.num_frames

    seg_file_dataset = SegFileDataset(
        unidet_outputs_dir=unidet_outputs_dir,
        video_dir= video_dir,
        annots_path=ego4d_annots_path
    )

    # for _, seg in tqdm.tqdm(enumerate(lta_annots)):

    #     vid_path = seg['clip_uid'] + ".mp4"
    #     start_frame = seg['action_clip_start_frame']
    #     end_frame = seg['action_clip_end_frame']

    #     seg_name = vid_path.split(".m")[0] + "_start_frame_" + str(start_frame) +\
    #           "_end_frame_" + str(end_frame)

    #     # Load unidet outputs for the segment
    #     with open(unidet_outputs_dir + seg_name + ".json", "r") as f:
    #         unidet_out = json.load(f)        
        
    #     frame_ids = list(unidet_out.keys()) # 16 frame ids

    #     # (num_frames, H, W, C) Convert to torch format.
    #     frames = video_loader_by_frames(args.video_dir, vid_path, frame_ids) 

    #     # # (num_frames, H, W, C)
    #     # np_frames = convert_tensors_to_images(frames)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('extract Unidet outputs')
    parser.add_argument('--video-dir', default='assets/3c0dffd0-e38e-4643-bc48-\
                        d513943dc20b_012_014.mp4', type=str, help='video directory')
    parser.add_argument('--save-dir', default='assets/3c0dffd0-e38e-4643-bc48-d513943dc20b_012\
                        _014.mp4', type=str, help='path to save the generated segments')    
    parser.add_argument('--ego4d_annots_path', type=str, help='Path to Ego4D LTA annotations.')
    parser.add_argument('--num_frames', default=16, help="number of frames to be sampled.")
    parser.add_argument('--unidet_outputs_dir', type=str, help='Path to Unidet outputs')

    args = parser.parse_args()
    main(args)

    
