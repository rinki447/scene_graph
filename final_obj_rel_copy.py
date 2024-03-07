import glob
import json
import torch
import os
import numpy as np
import cv2
np.set_printoptions(precision=4)
import copy
import datetime
import time
from torchvision.transforms import transforms, InterpolationMode
from argparse import ArgumentParser
import torch.nn as nn
import torch.nn.functional as F

BATCHNORM_MOMENTUM = 0.01


#dataloader
from dataloader.AG_RELATION import AG_relations
from dataloader.action_genome import AG, cuda_collate_fn
from dataloader.base_dataset import BaseDatasetFrames
from dataloader.coco import COCO

# lib files
from lib.tempura import TEMPURA
from lib.draw_rectangles.draw_rectangles import draw_union_boxes

# lib2 files
from lib2.config import Config
from lib2.object_detector import detector
from lib2.sttran_sgdet_test_partA import STTran

#fasterRCNN files
from fasterRCNN.lib.model.faster_rcnn.resnet import resnet
from fasterRCNN.lib.model.utils.blob import prep_im_for_blob, im_list_to_blob
from fasterRCNN.lib.model.roi_layers import nms


##########################################################################################################################################
class Config(object):
	def __init__(self):
		"""Defaults"""
		self.mode = None
		self.save_path = None
		self.model_path = None
		self.data_path = None
		#self.input_dir=None
		self.original_video_path=None
		self.modified_file_path=None
		self.saving_dir_path=None
		self.output_dir=None
		self.datasize = None
		self.ckpt = None
		self.optimizer = None
		self.bce_loss = None
		self.lr = 1e-5
		self.enc_layer = 1
		self.dec_layer = 3
		self.nepoch = 10
		self.parser = self.setup_parser()
		self.args = vars(self.parser.parse_args())
		self.__dict__.update(self.args)
		
		if self.mem_feat_lambda is not None:
			self.mem_feat_lambda = float(self.mem_feat_lambda)
		
		
		if self.rel_mem_compute == 'None' :
			self.rel_mem_compute = None
		if self.obj_loss_weighting == 'None':
			self.obj_loss_weighting = None
		if self.rel_loss_weighting == 'None':
			self.rel_loss_weighting = None

	def setup_parser(self):
		"""Sets up an argument parser:return:"""
		parser = ArgumentParser(description='training code')
		parser.add_argument('-mode', dest='mode', help='predcls/sgcls/sgdet', default='predcls', type=str)
		parser.add_argument('-save_path', default=None, type=str)
		parser.add_argument('-model_path', default=None, type=str)
		parser.add_argument('-data_path', default='data/ag/', type=str)
		#parser.add_argument('-input_dir', default=None, type=str)
		parser.add_argument('-original_video_path', default=None, type=str)
		parser.add_argument('-modified_file_path', default=None, type=str)
		parser.add_argument('-output_dir', default=None, type=str)
		parser.add_argument('-saving_dir_path', default=None, type=str)

		parser.add_argument('-datasize', dest='datasize', help='mini dataset or whole', default='large', type=str)
		parser.add_argument('-ckpt', dest='ckpt', help='checkpoint', default=None, type=str)
		parser.add_argument('-optimizer', help='adamw/adam/sgd', default='adamw', type=str)
		parser.add_argument('-lr', dest='lr', help='learning rate', default=1e-5, type=float)
		parser.add_argument('-nepoch', help='epoch number', default=10, type=int)
		parser.add_argument('-enc_layer', dest='enc_layer', help='spatial encoder layer', default=1, type=int)
		parser.add_argument('-dec_layer', dest='dec_layer', help='temporal decoder layer', default=3, type=int)

		#logging arguments
		parser.add_argument('-log_iter', default=100, type=int)
		parser.add_argument('-no_logging', action='store_true')
 
		# heads arguments
		parser.add_argument('-obj_head', default='gmm', type=str, help='classification head type')
		parser.add_argument('-rel_head', default='gmm', type=str, help='classification head type')
		parser.add_argument('-K', default=4, type=int, help='number of mixture models')

		# tracking arguments
		parser.add_argument('-tracking', action='store_true')

		# memory arguments
		parser.add_argument('-rel_mem_compute', default=None, type=str, help='compute relation memory hallucination [seperate/joint/None]')
		parser.add_argument('-obj_mem_compute', action='store_true')
		parser.add_argument('-take_obj_mem_feat', action='store_true')
		parser.add_argument('-obj_mem_weight_type',default='simple', type=str, help='type of memory [both/al/ep/simple]')
		parser.add_argument('-rel_mem_weight_type',default='simple', type=str, help='type of memory [both/al/ep/simple]')
		parser.add_argument('-mem_fusion',default='early', type=str, help='early/late')
		parser.add_argument('-mem_feat_selection',default='manual', type=str, help='manual/automated')
		parser.add_argument('-mem_feat_lambda',default=None, type=str, help='selection lambda')
		parser.add_argument('-pseudo_thresh', default=7, type=int, help='pseudo label threshold')

		# uncertainty arguments
		parser.add_argument('-obj_unc', action='store_true')
		parser.add_argument('-rel_unc', action='store_true')

		#loss arguments
		parser.add_argument('-obj_loss_weighting',default=None, type=str, help='ep/al/None')
		parser.add_argument('-rel_loss_weighting',default=None, type=str, help='ep/al/None')
		parser.add_argument('-mlm', action='store_true')
		parser.add_argument('-eos_coef',default=1,type=float,help='background class scaling in ce or nll loss')
		parser.add_argument('-obj_con_loss', default=None, type=str,  help='intra video visual consistency loss for objects (euc_con/info_nce)')
		parser.add_argument('-lambda_con', default=1,type=float,help='visual consistency loss coef')
		return parser


class object_saver():
	def obj_save(original_video_path,data_path,modified_file_path,saving_dir_path,mode):

		# sorting the original video paths and saving freshly in other file
		path= original_video_path 
		path2= modified_file_path # initially aimed to sort original path list alphabetically and save into modified one, but did not perform that
		
		with open(path,'r') as f:
			data_path_video = f.read().splitlines()

		dir1=[] 

		for i in data_path_video:
			dir1.append(i)

		sorted_paths_str1= '\n'.join(dir1) #sorting steps were not done

		with open(path2,'w') as f:
			f.write(sorted_paths_str1) 

		############################################################################################################################

		""" creating transformed image dataset, dataloader"""

		max_frames=256
		normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
		transform = transforms.Compose([transforms.Resize((384,384),interpolation=InterpolationMode.BICUBIC),transforms.ToTensor(),normalize]) 
		dataset=BaseDatasetFrames(modified_file_path,transform,max_frames)
		dataloader = torch.utils.data.DataLoader(dataset, shuffle=False, num_workers=1, collate_fn=cuda_collate_fn)

		##################################################################################################################
		# creating Action Genome dataloader to get list of its object classes and relationship classes

		#AG_dataset = AG(mode="test", datasize="large", data_path=data_path, filter_nonperson_box_frame=True,
				#filter_small_box=False if mode == 'predcls' else True)
		
		COCO_dataset=COCO(mode="test", datasize="large", data_path=data_path, filter_nonperson_box_frame=True,
				filter_small_box=True) #) if mode == 'predcls' else True)
		###########################################################################################################################

		gpu_device = torch.device('cuda:0')
		
		##################### initialize object detector with faster RCNN model checkpoint ##################################################33
		
		checkpoint = torch.load('fasterRCNN/models/faster_rcnn_coco.pth') 
		object_detector = detector(checkpoint=checkpoint, train=False, object_classes=COCO_dataset.object_classes, use_SUPPLY=True, mode=mode).to(device=gpu_device)
		object_detector.eval()

		print("dataloader lngth or b length:",len(dataloader))

		########################################################################################################

		## for each vido save deteceted bject detials in evluation mode
		with torch.no_grad():
				
			for b, data in enumerate(dataloader):
				#for b in range(1758,4000): if certain videos to be processed
				#frames,vid_id,sampled_frame_ids = data[0],data[1],data[2]
				print(f"video {b} processing")
				v_id = copy.deepcopy(data[5])
				f_path = copy.deepcopy(data[6])
				path_list = copy.deepcopy(data[7])
				
				print("v_id:",v_id)
				print("frame path for first frame in this video",path_list)
				print("before object_detector, no of frame in this video",len(f_path))
				#print("*********************************************************************")
				
				im_data = copy.deepcopy(data[0].cuda(0))
				im_info = copy.deepcopy(data[1].cuda(0))
				gt_boxes = copy.deepcopy(data[2].cuda(0))
				num_boxes = copy.deepcopy(data[3].cuda(0))
				gt_annotation = None 
				frame_no=im_data.shape[0]  

				# creating a entry dictionary containing details of detected box for all frames in a video
				entry = object_detector(im_data, im_info, gt_boxes, num_boxes, gt_annotation, im_all=None)
				frames_before_nms=np.unique(entry['boxes'][:, 0].cpu().numpy())
				print("after object detector, before nms total unique frames are",len(frames_before_nms))

				# save feature map for all frames of a video in a single .npz file
				kept_frame_index=np.unique(entry["boxes"][:,0].cpu().numpy())
				kept=torch.tensor(kept_frame_index).long()
				feats_fmap= entry['fmaps'][kept].cpu().numpy()
				directory_path3= os.path.join(conf.saving_dir_path,f"{v_id}")
				os.makedirs(directory_path3, exist_ok=True)    
				custom_path3 = os.path.join(directory_path3,"npz_fmap.npz")	
				np.savez(custom_path3, data_fmap=feats_fmap)
				
				##############'''Code for clean class and NMS application'''################
				box_idx = entry['boxes'][:, 0].long()
				b1 = int(box_idx[-1] + 1)
				
				# # NMS
				final_boxes = []
				final_dists = []
				final_feats = []

				print("before nms no of boxes",entry['boxes'].shape[0])

				for i in range(b1): # for all probable frames in a video
					# images in the batch

					scores = entry['distribution'][entry['boxes'][:, 0] == i]
					pred_boxes = entry['boxes'][entry['boxes'][:, 0] == i, 1:]
					feats = entry['features'][entry['boxes'][:, 0] == i]
					no=pred_boxes.shape[0] 
					########################################### take those frames only, which includes boxes#######################

					if no>0: #for frames where atleat one objet is detected
						for j in range(len(COCO_dataset.object_classes) - 1):
							# NMS according to obj categories
							inds = torch.nonzero(torch.argmax(scores, dim=1) == j).view(-1)
							# if there is det
							if inds.numel() > 0:
								cls_dists = scores[inds]
								cls_feats = feats[inds]
								cls_scores = cls_dists[:, j]
								_, order = torch.sort(cls_scores, 0, True)
								cls_boxes = pred_boxes[inds]
								cls_dists = cls_dists[order]
								cls_feats = cls_feats[order]
								keep = nms(cls_boxes[order, :], cls_scores[order], 0.6)  # hyperparameter

								final_dists.append(cls_dists[keep.view(-1).long()])
								final_boxes.append(torch.cat((torch.tensor([[i]], dtype=torch.float).repeat(keep.shape[0],
																								1).cuda(0),
												  cls_boxes[order, :][keep.view(-1).long()]), 1))
								final_feats.append(cls_feats[keep.view(-1).long()])
		

				
				entry['boxes'] = torch.cat(final_boxes, dim=0)
				box_idx = entry['boxes'][:, 0].long()
				entry['distribution'] = torch.cat(final_dists, dim=0)
				entry['features'] = torch.cat(final_feats, dim=0)
				entry['pred_scores'], entry['pred_labels'] = torch.max(entry['distribution'], dim=1)
				
				print("after nms no of bbox reduced to:",entry['boxes'].shape[0])
				new_frame_no=np.unique(entry['boxes'][:,0].cpu().numpy())
				print("after obj detect, and nms total no of unique frame:",len(new_frame_no))
		

				##################### save .npz file for ROI features per frame  ##############################################################################
		
				#find exact frame numbers in a video
				b=int(b)

				frames=np.unique(entry['boxes'][:, 0].cpu().numpy())
		
				print("finally total no of frames are",len(frames))

				c=0
				for frame_id in frames:
					f_id=int(frame_id) 
					feats_frame= entry['features'][entry['boxes'][:, 0] == int(frame_id)]
					b_l=feats_frame.shape[0]
		
					#for each box in that frame: this loop helps to create custom key
					final_array={}
					for box_no in range(b_l):
						array = feats_frame[box_no].cpu().numpy()
						final_array[f"box:no:{box_no}"]=array ####### index=box_no

					directory_path2= os.path.join(saving_dir_path,f"{v_id}/{f_path[f_id]}")
					os.makedirs(f"{directory_path2}", exist_ok=True)
					custom_path1= os.path.join(directory_path2,"npz_file.npz")
					np.savez(custom_path1, **final_array, allow_pickle=True)	
			

					##################example of how to load .npz file using key (not required ) ###########
					if c==0:
						loaded_data = np.load(custom_path1, allow_pickle=True) ###############  load npz file using a key to print
						ans=loaded_data["box:no:0"]   ################# change 0 with {box_no} variable used above
						#print("npz feature for box:0 for first frame",ans)
						loaded_data.close()
					########################################################################	
					c=c+1
					

			
				########## save .json files per frame for detected object details  #################################################################################
			
				data={}
				for frame_id in frames:
						f_id=int(frame_id)
						boxes=entry['boxes'][entry['boxes'][:,0]==frame_id]
						distributions=entry['distribution'][entry['boxes'][:,0]==frame_id]
						pred_scores=entry['pred_scores'][entry['boxes'][:,0]==frame_id]
						pred_labels=entry['pred_labels'][entry['boxes'][:,0]==frame_id]

						box_no=boxes.shape[0]
						box_data={}	

						for box in range(box_no):	
					
							box_data[f"box_id_{box}"] = {"box_detail": boxes[box,1:].cpu().numpy().tolist(),
							"distribution":distributions[box].cpu().numpy().tolist(),
							"pred_score": pred_scores[box].cpu().numpy().tolist(),
							"pred_label": pred_labels[box].cpu().numpy().tolist()}
				

						if f"{v_id}" not in data:
							data[f"{v_id}"] = {}

						if f"{f_path[f_id]}" not in data[f"{v_id}"]:    

							data[f"{v_id}"][f"{f_path[f_id]}"]= box_data     
							   
				directory_path1 = saving_dir_path
				os.makedirs(f"{directory_path1}", exist_ok=True) 
				os.makedirs(f"{directory_path1}/{v_id}", exist_ok=True)           
				custom_path1 = os.path.join(f"{directory_path1}/{v_id}", "Activity_BBox.json")
				with open(custom_path1, "w") as json_file:
						json.dump(data, json_file, indent=4)

				print(f"JSON file for video_index={v_id} for single video saved to {custom_path1}")
			
				###################################################################################################################	
			

				print(f"All steps done for video_{v_id}") 
				print("********************************************************************************") 
				
				#if (b==0):   	
					#exit()    	

		
			print("done for all videos")



class annotation():
		def gt_annotation(saved_json_path,key_vid,entry,frame_box):

			with open(saved_json_path,'r') as f: 
				json_file= json.load(f)    

			frame_list=list(json_file[f"{key_vid}"].keys())
			frame_name_list=np.unique(frame_box[:,0])
			uniq_frames=np.unique(entry["boxes"][:,0])
				   
			gt_annotation_video = []
			entry_frame={}
			
			
			for i in uniq_frames:####### each frame j

				ind=torch.nonzero(entry["boxes"][:,0]==i).view(1,-1)
				ind=ind.squeeze(0)
				
			
				entry_frame["boxes"]=entry["boxes"][ind]
				entry_frame["pred_scores"]= entry["pred_scores"][ind]
				entry_frame["distribution"]= entry["distribution"][ind]
				entry_frame["labels"]= entry["labels"][ind]

				indice=np.arange(entry_frame["boxes"].shape[0])
				h_idx=np.where(entry_frame["labels"]==0)


				if len(h_idx[0]) > 0: # if there is atleast one human box
					entry_hum=entry_frame["distribution"][h_idx[0]]
					human_idx_local = torch.argmax(entry_hum[:,0]) 
					human_idx=indice[h_idx][human_idx_local]
					
				else:

					human_idx = torch.argmax(entry_frame['distribution'][:,0])  # the local bbox index with highest human score in this frame
					

				human_idx=human_idx.item() 
				human_bbox=entry_frame["boxes"][human_idx,1:].cpu().numpy().tolist()
				human=[]
				human.append(human_bbox)

				obj_box_len=entry_frame["boxes"][:,0].shape[0]-1 # 1 of the boxes belongs to human
				box_idx=np.arange(0,(entry_frame["boxes"].shape[0])) #index of all boxes
				obj_box_idx = np.delete(box_idx, np.where(box_idx == int(human_idx))) # index of non-human boxes
							
				gt_annotation_frame = [{'person_bbox':np.array(human, dtype=np.float32), "frame": f"{i}"}] 

				obj_cls_list=entry_frame["labels"][obj_box_idx]

				c=0

				for k in obj_cls_list: 
					
					aa={}
					aa['class'] = int(k)
					id=obj_box_idx[c]
					aa['bbox'] = np.array(entry_frame['boxes'][id,1:].cpu().numpy().tolist())                      
					gt_annotation_frame.append(aa)
					c=c+1
								  
				gt_annotation_video.append(gt_annotation_frame) ##### frame wise append
						
			return(gt_annotation_video)



class load_saved_detection():
	def load_entry(saved_json_path,vid,original_video_path,fmap_path,directory_path):
		entry={}
		boxes=[]
		distribution=[]
		pred_scores=[]
		pred_labels=[]
		frame_box=[]
		frames=[]
		im_scales=[]

		# open json file, saved by object_saver.py, corresponding to each video
		with open(saved_json_path,'r') as f: 
			json_file= json.load(f)

		key_vid=vid     
		frame_list=list(json_file[f"{key_vid}"].keys())

		#*** start: extra step for image and im_scale (if not saved by object_saver.py)********
	
		# get original file path of the same video from text file, as saved by annotation.py
		with open(original_video_path,'r') as f: 
			video_path=f.read().splitlines() # save all video path
			for pth in video_path:
				if pth.split("/")[-1]==key_vid: 
					my_video_path=pth # save single (matched one) video path

		c=0
		for i,key_frame in enumerate(frame_list):

				frame_path=os.path.join(my_video_path,f"{key_frame}.jpg")
				image=cv2.imread(frame_path,cv2.IMREAD_UNCHANGED)
				im, im_scale = prep_im_for_blob(image, [[[102.9801, 115.9465, 122.7717]]], 384, 384) #adopted from AG_dataset of action_genome.py
				im_scales.append(im_scale)
				frames.append(im)
	  
				#**************************** ends extra steps for image and im scale ***************

				################## collecting saved box data for single video into entry dictionary #####################
	  
				box_list=list(json_file[f"{key_vid}"][f"{key_frame}"].keys())

				for key_box in box_list: ###### for all boxes in single frame
					frame_box.append([key_frame,int(i),int(c),key_box]) #frame_name, frame_number,box_global_number
	  
					box_coord=(json_file[f"{key_vid}"][f"{key_frame}"][f"{key_box}"]["box_detail"]) # "box_detail" keyword is used while saving
					box_coord.insert(0,int(i)) # frame number should be the first column in entry["boxes"] (this is not the original one saved during object detection)
					boxes.append(box_coord) # all boxes for all frames in single video are being saved in "boxes" list

					distribution.append(json_file[f"{key_vid}"][f"{key_frame}"][f"{key_box}"]["distribution"])
					pred_labels.append(json_file[f"{key_vid}"][f"{key_frame}"][f"{key_box}"]["pred_label"])
					pred_scores.append(json_file[f"{key_vid}"][f"{key_frame}"][f"{key_box}"]["pred_score"])

					c=c+1

	
		blob = im_list_to_blob(frames)     #frames= list of all images
		im_info = np.array([[blob.shape[1], blob.shape[2], im_scales[0]]],dtype=np.float32)  
		im_info = torch.from_numpy(im_info).repeat(blob.shape[0], 1)

		# info about all boxes from all frames of single video are saved in "entry"
		entry["boxes"]=torch.tensor(boxes)    
		entry["distribution"]=torch.tensor(distribution)  
		entry["labels"]=torch.tensor(pred_labels) 
		entry["pred_scores"]=torch.tensor(pred_scores)  
		entry["im_info"]=im_info

		# upload f_map for each frame
		f_fmap=fmap_path
		npz_file= np.load(f_fmap)
		entry["fmaps"]=torch.tensor(npz_file["data_fmap"]) # key to load data here is "data_fmap" as used while saved
  
		# upload features for each box in a frame and likewise for all frame
		features=[]
		for frame in frame_list:
			  
			f=os.path.join(directory_path,f'{vid}/{frame}/npz_file.npz') 
			npz_file= np.load(f)
			npz_length=len(npz_file)

			for i in range(npz_length-1): # "allow pickle" is also in the list at the end, omit that
				features.append(npz_file[f"box:no:{i}"]) #### that is how loading custom key was saved
	
		entry["features"]=torch.tensor(np.array(features))

		frame_box=np.array(frame_box) # list of frame_name, frame_number,box_global_number ,box_local_no for all box in a video
		return entry,frame_box



	############################# filter_no_human.py ###############################################

class filter_nonhuman():
	def filter(entry, frame_box):

		uniq_frames=np.unique(entry["boxes"][:,0])
		frame_id=[]
		index=torch.tensor([])
		entry_frame={}
		ent={}
		frame_no=0

		# find human/"person" boxes in each frame
		for i in uniq_frames:
	
			ind=torch.nonzero(entry["boxes"][:,0]==i).view(1,-1)
			indice=ind.squeeze(0)
			ent["labels"]= entry["labels"][indice]


			if 0 in ent["labels"]: # if there is atleast 1 "person" box in a frame, the frame is kept
				frame_id.append(frame_no)
				index=torch.cat((index,indice),0) # global index of that frame_boxes are saved
	  
			frame_no=frame_no+1

		index=index.squeeze(0).long() # list of global index of all boxes (belonging to filtered human frames) of a video 
	
		# box, score,distribution, label are changed for all boxes
	
		entry_frame["boxes"]=entry["boxes"][index]
		entry_frame["pred_scores"]= entry["pred_scores"][index]
		entry_frame["distribution"]= entry["distribution"][index]
		entry_frame["labels"]= entry["labels"][index] 
		entry_frame["features"]= entry["features"][index] 

		frame_id=torch.tensor(frame_id).long() # list of indices of all filtered human frames of a video 

		# fmap and im_info are changed as per frame basis

		entry_frame["fmaps"]=entry["fmaps"][frame_id] 
		entry_frame["im_info"]=entry["im_info"][frame_id]

		frame_box1=frame_box[index.cpu().numpy()] ### filtered list of (frame_name,frame_no,global_box_no,local_box_no)

		return entry_frame, frame_box1 


############################### non_max_suppression.py ######################################
class non_max():
	def nms(entry):
		nms_box=torch.tensor([])
		nms_scores=torch.tensor([])
		nms_distribution=torch.tensor([])
		nms_label=torch.tensor([])
		nms_feature=torch.tensor([])
		uniq_frames=np.unique(entry["boxes"][:,0].cpu().numpy())
		entry_idx=torch.tensor([])
		entry_idx = entry_idx.to(dtype=torch.long)
		idx1=torch.tensor(np.arange(0,entry["boxes"].shape[0]))

		sum=0
  
		for i in uniq_frames:

			idx2=idx1[entry["boxes"][:,0]==i].clone().detach()

			frame_box=entry["boxes"][entry["boxes"][:,0]==i]
			frame_scores=entry["pred_scores"][entry["boxes"][:,0]==i]
			frame_distribution=entry["distribution"][entry["boxes"][:,0]==i]
			frame_labels=entry["labels"][entry["boxes"][:,0]==i]
			frame_features=entry["features"][entry["boxes"][:,0]==i]
			uniq_labels=np.sort(np.unique(frame_labels))

			for j in uniq_labels:
				idx3=idx2[frame_labels==j].clone().detach()
	  
				class_box=frame_box[frame_labels==j]
				cls_box_list=class_box.cpu().numpy().tolist()
				class_scores=frame_scores[frame_labels==j]
				cls_scores_list=class_scores.cpu().numpy().tolist()
				class_distribution=frame_distribution[frame_labels==j]
				class_label=frame_labels[frame_labels==j]
				class_feature=frame_features[frame_labels==j]


				boxes_for_nms = [[x1, y1, x2, y2] for _,x1, y1, x2, y2 in cls_box_list]
				confidences = [confidence for confidence in cls_scores_list]

				keep = cv2.dnn.NMSBoxes(boxes_for_nms, confidences, 0.0, 0.5)

		

				nms_box=torch.cat((nms_box,class_box[keep]),0)
				nms_scores=torch.cat((nms_scores,class_scores[keep]),0)
				nms_distribution=torch.cat((nms_distribution,class_distribution[keep]),0)
				nms_label=torch.cat((nms_label,class_label[keep]),0)
				nms_feature=torch.cat((nms_feature,class_feature[keep]),0)
				idx4=idx3[keep].clone().detach().long()
	  
				entry_idx=torch.cat((entry_idx,idx4),0)
	  

		entry["boxes"]=nms_box
		entry["pred_scores"]=nms_scores
		entry["distribution"]=nms_distribution
		entry["labels"]=nms_label
		entry["features"]=nms_feature

		print("total box after nms",entry["boxes"].shape[0])

		return entry,entry_idx       

############################## pair.py #####################################################

class pair_maker(nn.Module):

	def __init__(self, train, object_classes, use_SUPPLY, mode='predcls'):
		super(pair_maker, self).__init__()

		self.use_SUPPLY = use_SUPPLY
		self.object_classes = object_classes
		self.mode = mode
		self.fasterRCNN = resnet(classes=self.object_classes, num_layers=101, pretrained=False, class_agnostic=False)
		self.fasterRCNN.create_architecture()
		checkpoint = torch.load('fasterRCNN/models/faster_rcnn_coco.pth')
		self.fasterRCNN.load_state_dict(checkpoint['model'])
		self.ROI_Align = copy.deepcopy(self.fasterRCNN.RCNN_roi_align)
		self.RCNN_Head = copy.deepcopy(self.fasterRCNN._head_to_tail)

	def forward(self,gt_annotation,entry):

			im_info=entry["im_info"]
			# how many bboxes we have
			bbox_num = 0
			im_idx = []  # which frame are the relations belong to
			pair = []

			for i in gt_annotation:
				bbox_num += len(i)
				
			FINAL_BBOXES = torch.zeros([bbox_num,5], dtype=torch.float32).cuda(0)
			FINAL_LABELS = torch.zeros([bbox_num], dtype=torch.int64).cuda(0)
			FINAL_SCORES = torch.ones([bbox_num], dtype=torch.float32).cuda(0)
			HUMAN_IDX = torch.zeros([len(gt_annotation),1], dtype=torch.int64).cuda(0)

			bbox_idx = 0
			for i, j in enumerate(gt_annotation):
				for m in j:
					if 'person_bbox' in m.keys():
						FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['person_bbox'][0])
						FINAL_BBOXES[bbox_idx, 0] = i
						FINAL_LABELS[bbox_idx] = 0#?????????????????????????????
						HUMAN_IDX[i] = bbox_idx
						bbox_idx += 1
					else:
						FINAL_BBOXES[bbox_idx,1:] = torch.from_numpy(m['bbox'])
						FINAL_BBOXES[bbox_idx, 0] = i
						FINAL_LABELS[bbox_idx] = m['class']
						im_idx.append(i)
						pair.append([int(HUMAN_IDX[i]), bbox_idx]) # saves local_human_id of the frame
						bbox_idx += 1
			pair = torch.tensor(pair).cuda(0)
			im_idx = torch.tensor(im_idx, dtype=torch.float).cuda(0)

			counter = 0
			FINAL_BASE_FEATURES = torch.tensor([]).cuda(0)
			
			#use entry["fmaps"] for each video as FINAL_BASE_FEATURES
			FINAL_BASE_FEATURES =entry["fmaps"]
			###################################################################################    
		  
			FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] * im_info[0, 2]
			FINAL_BBOXES = copy.deepcopy(FINAL_BBOXES.cuda(0))
			FINAL_BASE_FEATURES = copy.deepcopy(FINAL_BASE_FEATURES.cuda(0))
			FINAL_FEATURES=entry["features"].cuda(0)

			if self.mode == 'predcls':

				union_boxes = torch.cat((im_idx[:, None], torch.min(FINAL_BBOXES[:, 1:3][pair[:, 0]], FINAL_BBOXES[:, 1:3][pair[:, 1]]),
										 torch.max(FINAL_BBOXES[:, 3:5][pair[:, 0]], FINAL_BBOXES[:, 3:5][pair[:, 1]])), 1)
				union_feat = self.fasterRCNN.RCNN_roi_align(FINAL_BASE_FEATURES, union_boxes)
				FINAL_BBOXES[:, 1:] = FINAL_BBOXES[:, 1:] / im_info[0, 2]
				pair_rois = torch.cat((FINAL_BBOXES[pair[:, 0], 1:], FINAL_BBOXES[pair[:, 1], 1:]),
									  1).data.cpu().numpy()
				spatial_masks = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(FINAL_FEATURES.device)

				entry["im_idx"]=im_idx
				entry['pair_idx']= pair
				entry['human_idx']=HUMAN_IDX
				entry['union_feat']= union_feat
				entry['union_box']= union_boxes
				entry['spatial_masks']= spatial_masks
						
					  

			return entry


	############################################################################################



def main(conf):
		#conf = Config()
		conf.datasize="large"
		gpu_device = torch.device('cuda:0')
	 
		data_path=conf.data_path ######################## used for AG and coco only
		modified_data_path=conf.modified_file_path ################## sorted_video_path
		saving_dir_path=conf.saving_dir_path ####################### json file path to save detected object details
		original_video_path=conf.original_video_path
		mode=conf.mode
		mode_obj="sgdet"
		
	
		object_saver.obj_save(original_video_path,data_path,modified_data_path,saving_dir_path,mode_obj)

		#AG_dataset = AG(mode="test", datasize=conf.datasize, data_path=data_path, filter_nonperson_box_frame=True,
			#filter_small_box=False if conf.mode == 'predcls' else True)
		AG_rel_classes=AG_relations(mode="test", datasize=conf.datasize, data_path=data_path, filter_nonperson_box_frame=True,
			filter_small_box=False if conf.mode == 'predcls' else True)  ##### rpossible elations classes taken from AG
		COCO_dataset = COCO(mode="test", datasize=conf.datasize, data_path=data_path, filter_nonperson_box_frame=True,
			filter_small_box=False if conf.mode == 'predcls' else True)   #### obj_classes taken from coco
	
		##### collecte video names from data generated by object_saver.py
		directory_path = conf.saving_dir_path ######## collect object details from saved folder
	
		all_v_idx=next(os.walk(directory_path))[1]
	
		gpu_device = torch.device('cuda:0')
	
		pairing=pair_maker(train=False,object_classes=COCO_dataset.object_classes, use_SUPPLY=True, mode='predcls').to(device=gpu_device)
	
	
		model = TEMPURA(mode="predcls",
				   attention_class_num=len(AG_rel_classes.attention_relationships),
				   spatial_class_num=len(AG_rel_classes.spatial_relationships),
				   contact_class_num=len(AG_rel_classes.contacting_relationships),
				   obj_classes=COCO_dataset.object_classes,
				   enc_layer_num=conf.enc_layer,
				   dec_layer_num=conf.dec_layer,
				   obj_mem_compute = conf.obj_mem_compute,
				   rel_mem_compute = conf.rel_mem_compute,
				   take_obj_mem_feat= conf.take_obj_mem_feat,
				   mem_fusion= conf.mem_fusion,
				   selection = conf.mem_feat_selection,
				   selection_lambda=conf.mem_feat_lambda,
				   obj_head = conf.obj_head,
				   rel_head = conf.rel_head,
				   K = conf.K,
				   tracking= conf.tracking).to(device=gpu_device)
	
		model.eval()
	 
	
		ckpt = torch.load(conf.model_path, map_location=gpu_device)
		ckpt_clone = ckpt['state_dict'].copy()
	
	
		for k in list(ckpt['state_dict'].keys()):
			if 'object_classifier' in k or 'obj_embed' in k:
	
				ckpt_clone.pop(k)
	
		model.load_state_dict(ckpt_clone, strict=False) 
	
		print('*'*50)
		print('CKPT {} is loaded'.format(conf.model_path))
	
		count=0
	
		for vid in list(all_v_idx) : 
			#vid="v_biyf6Q-xF0M"
			data={}
	  
	
			saved_json_path=os.path.join(saving_dir_path,f"{vid}/Activity_BBox.json")   
			saved_fmap_path=os.path.join(saving_dir_path,f"{vid}/npz_fmap.npz")  
			original_video_path=conf.original_video_path
			####### load saved entry boxes, labels, distributions, score, fmap, im_info #####
	
			#frame_box1=[frame_name, frame_number,box_global_number] for all box
			entry,frame_box1=load_saved_detection.load_entry(saved_json_path,vid,original_video_path,saved_fmap_path,saving_dir_path) 
			unq_f1=np.unique(entry["boxes"][:,0].cpu().numpy())
			b_n1=entry["boxes"][:,0].cpu().numpy().shape
			print("no of frame after loading",len(unq_f1))
			print("no of box after loading",b_n1)
			print("no of box after loading from frame_box1",frame_box1.shape)
			print("entry frames after data load",entry["boxes"][:,0].cpu().numpy())
			#	print(frame_box1[0:37])
	  
	  
	
			######################  filter non human frames #################
	
			# frame_box11= list of global index of the boxes filtered
			entry, frame_box11=filter_nonhuman.filter(entry, frame_box1)
	  
			unq_f2=np.unique(entry["boxes"][:,0].cpu().numpy())
			b_n2=entry["boxes"][:,0].cpu().numpy().shape
			print("no of frame after discarding non human frames",len(unq_f2))
			print("no of box after discarding non human frames",b_n2)
			print("no of box after discarding non human frames from frame_box11",frame_box11.shape)
			print("entry frames after discarding non human frames",entry["boxes"][:,0].cpu().numpy())
			#print(frame_box11[0:37])
	  
			#print(frame_box11)
			
			################ apply nms ##################################
	 
			entry,entry_idx=non_max.nms(entry)
	   
			frame_box111=frame_box11[entry_idx.cpu().numpy()]
			unq_f3=np.unique(entry["boxes"][:,0].cpu().numpy())
			b_n3=entry["boxes"][:,0].cpu().numpy().shape

			print("no of frame after nms",len(unq_f3))
			print("no of box after nms",b_n3)
			print("no of box after nms from frame_box111",frame_box111.shape)
			print("entry frames after nms",entry["boxes"][:,0].cpu().numpy())
	  

	 
			# im_info and fmap are related to each image, not boxes of the image, so nms does not change them
			
			############################### create gt_annotation ###########################################################
	  
			gt_annotation_video=annotation.gt_annotation(saved_json_path,vid,entry,frame_box111)
			gt_annotations=gt_annotation_video
	
			bbox_num=0
			for i in gt_annotations:
				bbox_num += len(i)
			print("no of box after gt annotation",bbox_num)
			print("entry frames after gt_annotation",entry["boxes"][:,0].cpu().numpy())
	  
			############################### saving all frame names and paths in a video #######################################################
	
	
			with open(original_video_path,'r') as f: ##### change for dataset, train/test case
				video_path=f.read().splitlines() ####save all video path
			for i,pth in enumerate(video_path):
				if pth.split("/")[-1]==vid: 
					my_video_path=pth 
	
			with open(saved_json_path,'r') as f:  #####change for dataset, train/test video case
				json_file= json.load(f)
	
			###### getting frame names ##################        
	  
			frame_list=np.unique(frame_box111[:,0])
	  
			print("frame list length", frame_list.shape)
	
	  
			frame_paths=[]
			for kk in frame_list:
		
				frame_paths.append(os.path.join(my_video_path, f'{kk}.jpg')) 
			print("frame path length",len(frame_paths))  
	  
	  
			##### human_object pairing (following predcls part of object_detector.py)#####################################
	
			im_info=entry["im_info"]
			entry=pairing(gt_annotations, entry)
			print("entry frames after pairing, before tempura",entry["boxes"][:,0].cpu().numpy())

			print("total no of box after pairing, before tempura",entry["boxes"].cpu().numpy().shape)

			print("no of human subject box after pairing, before tempura",entry["human_idx"].cpu().numpy().shape)
			print("pair idx shape after pairing, before tempura",entry["pair_idx"].cpu().numpy().shape)
			print("pair idx after pairing, before tempura",entry["pair_idx"])
			#if conf.tracking:
				#get_sequence(entry, gt_annotations, (im_info[0][:2]/im_info[0,2]).cpu().data,conf.mode)
	
	
			####### relationship prediction for predcls tempura test mode ###################################################
	
	
			pred = model(entry,phase='test', unc=False) 
			print("pair idx after tempura",pred["pair_idx"])
			print("pair idx shape after tempura",pred["pair_idx"].shape)
			############################## print objects and relations per frame ###########################################    

	  
					   
			row=[]
			prev_row_n=0
			a_rel_frame=[]
			id_frame=np.unique(pred["boxes"][:,0].cpu().numpy())

			frame_list=np.unique(frame_box111[:,0])

			#print(id_frame)
			#print(list(frame_box111[:,1]))
	  
			integer_list = [int(x) for x in frame_box111[:,1]]
			frame_idx_list=np.unique(integer_list)

			c3=0
			#frame_list=np.unique(entry["boxes"][:,0])
			for i,key_frame in enumerate(frame_list):
		
				#ind=torch.nonzero[entry["boxes"][:,0]==i]
				#key_frame=frame_box111[ind]
				#print(ind)
				#print(key_frame)

		
				rel_frame=[]
				all_box_idx=torch.nonzero(entry["boxes"][:,0]==frame_idx_list[i]).view(-1)  
				#print(f"no of box in frame {i}",all_box_idx.cpu().numpy().shape)
				#print(entry["pred_labels"][all_box_idx])
				#print("all_box_idx test print",all_box_idx)
				#print(len(all_box_idx))
		
				object_box_n=len(all_box_idx)-1
				hum_box_id=pred["human_idx"][i]
		
				if row==[]:
					prev_row_n=0
				else:
					prev_row_n=np.sum(row)
	
				row.append(object_box_n)
				end_row_n=np.sum(row)
	
		
				a_rel_frame=[]
				c_rel_frame=[]
				s_rel_frame=[]
	
				#not necessary step
				frame_path=os.path.join(my_video_path,f"{key_frame}.jpg")
				#print(f"frame_path:{frame_path}")
		
				#print("frame box shape",frame_box111.shape)
				for j in range(prev_row_n,end_row_n): ## all boxes in single frame
					rel_data={}
					a_rel_class=[]
					#hum=entry["pred_labels"][entry["pair_idx"][j,0]]
					hum_class=COCO_dataset.object_classes[1] ##### all human labels are marked as "80"
					obj=pred["pred_labels"][pred["pair_idx"][j,1]]
					obj = int(obj.item())
					obj_class=COCO_dataset.object_classes[obj+1]
		
					values_a = pred["attention_distribution"][j, :].detach().cpu().numpy()
					top_one_indices_a = list(np.argsort(values_a)[::-1][:1])
					elements_a = [AG_rel_classes.attention_relationships[i] for i in top_one_indices_a]
			   
	
					values_c = pred["contacting_distribution"][j, :].detach().cpu().numpy()
					top_three_indices_c = np.argsort(values_c)[::-1][:3]
					elements_c = [AG_rel_classes.contacting_relationships[i] for i in top_three_indices_c]
	
					values_s = pred["spatial_distribution"][j, :].detach().cpu().numpy()
					top_three_indices_s = np.argsort(values_s)[::-1][:3]
					elements_s = [AG_rel_classes.spatial_relationships[i] for i in top_three_indices_s]
		 
		  
					print(f"caption:{hum_class},{elements_a}/{elements_c}/{elements_s},{obj_class}")
					rel_data["subject"]=[hum_class,frame_box111[pred["pair_idx"][j,0].cpu().numpy().item(),3]]
					rel_data["object"]=[obj_class,frame_box111[pred["pair_idx"][j,1].cpu().numpy().item(),3]]
					rel_data["relation"]=[elements_a[0],elements_c[0],elements_c[1],elements_c[2],elements_s[0],elements_s[1],elements_s[2]]
		  
					rel_frame.append(rel_data)  


				if object_box_n==0:#only person detected in the frame , not other object detected
					rel_data={}
					hum_class=COCO_dataset.object_classes[1]
					rel_data["subject"]=[hum_class]
					rel_data["object"]=["none"]
					rel_data["relation"]=["none"]
					#print("frame with only 1 human",key_frame)
					rel_frame.append(rel_data)
	
				if f"{vid}" not in data:
					data[f"{vid}"] = {}

				if f"{key_frame}" not in data[f"{vid}"]:    

					data[f"{vid}"][f"{key_frame}"]= rel_frame 

				#directory_path1 = "/data/AmitRoyChowdhury/Rinki/Activity_test_relation"
				# Make sure the directory exists or create it if it doesn't
				os.makedirs(f"{conf.output_dir}", exist_ok=True) 
				os.makedirs(f"{conf.output_dir}/{vid}", exist_ok=True)           
				custom_path1 = os.path.join(f"{conf.output_dir}/{vid}", "Activity_relation.json")
				with open(custom_path1, "w") as json_file:
					json.dump(data, json_file, indent=4) 

				

			print(f"done for {count} video")
			count=count+1
			
		print("done for all videos")
				

			
	  
	  
			 
#total_time = time.time() - start_time
	
#total_time_str = str(datetime.timedelta(seconds=int(total_time)))
#print('Inference time {}'.format(total_time_str), flush=True)
# if conf.output_dir is not None:
#     with open(conf.output_dir+"log_"+conf.mode+".txt", "a") as f:
#                 f.truncate(0)
#                 f.close()
'''constraint_type = 'with constraint'
print('-'*10+constraint_type+'-'*10)
evaluator1.print_stats(log_file=log_val)
	
constraint_type = 'semi constraint'
print('-'*10+constraint_type+'-'*10)
evaluator2.print_stats(log_file=log_val)
	
constraint_type = 'no constraint'
print('-'*10+constraint_type+'-'*10)
evaluator3.print_stats(log_file=log_val)'''



if __name__ == '__main__':
	print("###########################")
	print("Calling main()")
	print("###########################")
	conf = Config()
	print("Calling config()")
	print("###########################")
	main(conf)