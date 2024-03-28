import torchvision.transforms as T
import numpy as np
import decord
import os.path as osp
import torch

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


def video_loader_by_frames(root, vid, frame_ids):
    vr = decord.VideoReader(osp.join(root, vid))
    try:
        frames = vr.get_batch(frame_ids).asnumpy()
        
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", vid)
        frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]

    #how to get im_info using im_scale and blob shape?

    img_tensor = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    img_tensor = img_tensor.permute(0, 3, 1, 2)
    gt_boxes = torch.zeros([img_tensor.shape[0], 1, 5])
    num_boxes = torch.zeros([img_tensor.shape[0]], dtype=torch.int64)

    return img_tensor, im_info, gt_boxes, num_boxes, frame_ids, vid
    #return torch.stack(frames, dim=0)
