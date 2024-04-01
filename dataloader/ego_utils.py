import torchvision.transforms as T
import numpy as np
import decord
import os.path as osp
import torch
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob


def convert_tensors_to_images(tensor_frames):
    """Given a torch tensor of images, convert to PIL format.    

       Args:
            tensor_frames: (N, H, W, C) - N frames in torch tensor format having (H, W, C) shape

    """
    pil_transform = T.ToPILImage()

    if tensor_frames.shape[1] > 4:
        tensor_frames = tensor_frames.permute(0, 3, 1, 2) # (N, C, H, W)

    pil_frames = []
    for frame in tensor_frames:
        pil_img = pil_transform(frame)
        pil_frames.append(pil_img)
    
    return pil_frames

def get_frame_ids(start_frame, end_frame, num_segments=32, jitter=True):
    seg_size = float(end_frame - start_frame - 1) / num_segments
    seq = []
    for i in range(num_segments):
        start = int(np.round(seg_size * i) + start_frame)
        end = int(np.round(seg_size * (i + 1)) + start_frame)
        end = min(end, end_frame)
        if jitter:
            frame_id = np.random.randint(low=start, high=(end + 1))
        else:
            frame_id = (start + end) // 2
        seq.append(frame_id)
    return seq


def get_image_scales(self, frames, frame_size=384):
        proc_frames = []
        img_scales = []
        for image in frames:
            img, img_scale = prep_im_for_blob(image, [[[102.9801, 115.9465, 122.7717]]], 
                                              frame_size, frame_size)
            proc_frames.append(img)
            img_scales.append(img_scale)

        return proc_frames, img_scales


def video_loader_by_frames(root, vid, frame_ids):
    vr = decord.VideoReader(osp.join(root, vid))
    try:
        frames = vr.get_batch(frame_ids).asnumpy()
        
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", vid)
        frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]


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

	return img_tensor, im_info, gt_boxes, num_boxes, frame_ids, vid
    #return torch.stack(frames, dim=0)