# Scene_graph

# Requirements
* python=3.6
* pytorch=1.1
* scipy=1.1.0
* cypthon
* dill
* easydict
* h5py
* opencv
* pandas
* tqdm
* yaml

# Steps

The list of path to all video frames of required dataset are kept in "original_video_path" file (text file). The "modified_file_path" is a blank text file where video paths will be saved in sorted manner by final_obj_rel_copy.py. The main file i.e, final_obj_rel_copy.py file loads those video frames using their path from this text file and then it initially uses fasterRCNN (coco trained) model to detect objects for all frames in all videos and save them into "saving_dir_path". 

Next the final_obj_rel_copy.py file accesses those object details using "saving_dir_path" and generates scene graph captions by using "tempura_predcls" model. The relationships detected between human-object pairs for each frame are saved in "output_dir" path.

"data_path" is used to access the text files containing the object classes of COCO dataset (on which fasterRCNN is trained for this case),and relationship classes of action genome dataset respectively, saved inside annotation folder.

1. Within scene_graph folder create "data" folder and download glove file there from https://drive.google.com/drive/folders/1Qrez_hfAKRqCqe7LlGX1xnGRwQZTdLVu?usp=sharing.
2. Create fasterRCNN folder within scene_graph and download the required (coco trained or action genome trained) Faster-RCNN model from https://github.com/jwyang/faster-rcnn.pytorch.
3. Set up the fasterRCNN trained on coco dataset:

```python
cd ~/scene_graph  
cd fasterRCNN
cd lib
python setup.py build develop
```
4. Download the tempura models (predcls) and save in "model_path" from https://drive.google.com/drive/folders/1m1xSUbqBELpogHRl_4J3ED7tlyp3ebv8
   
6. Find the required folders in this order within scene_graph folder:
* annotation
* data
* fasterRCNN
* dataloader
* lib
* lib2


7. Run the main file to generate scene graph from videos:
```python
python final_obj_rel_copy.py -mode predcls -datasize large -data_path ABC  -model_path XYZ.tar  -original_video_path MNP.txt -modified_file_path PQR.txt -output_dir  Activity_test_relation -saving_dir_path BCD -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.5  -rel_head gmm -obj_head linear -K 6 
```
