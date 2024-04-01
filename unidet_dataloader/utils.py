import torchvision.transforms as T
import numpy as np
import json
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

def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in xrange(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)
    return blob

def prep_im_for_blob(im, pixel_means, target_size, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)

    return im, im_scale

def video_loader_by_frames(root, vid, frame_ids):
    vr = decord.VideoReader(osp.join(root, vid))
    try:
        frames = vr.get_batch(frame_ids).asnumpy()
        frames = [torch.tensor(frame, dtype=torch.float32) for frame in frames]
    except (IndexError, decord.DECORDError) as error:
        print(error)
        print("Erroneous video: ", vid)
        frames = [torch.zeros((240, 320, 3)) for _ in range(len(frame_ids))]
    return torch.stack(frames, dim=0)

def return_id(unidet_label_file_path:str, class_name:str):
    """Given the unidet_label_file_path and a class_name, return the corresponding index.

    Args:
        unidet_label_file_path: (str) Path to the unidet label file "learned_mAP%2BM.json"
        class_name: (str) Class whose corresponding index you want to return.
    """
    with open(unidet_label_file_path, "r") as f:
        unidet_label_dict = json.load(f)

    for label_mapping in unidet_label_dict['categories']:
        if label_mapping['name'] == class_name:
            return label_mapping['id']
    