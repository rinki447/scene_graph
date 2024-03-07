# Scene_graph
The list of path to all video frames of required dataset are kept in "original_video_path" file (text file). The main file i.e, final_obj_rel_copy.py file loads those video frames using their path from this text file and then it initially uses fasterRCNN (coco trained) model to detect objects for all frames in all videos and save them into "saving_dir_path". 

Next the final_obj_rel_copy.py file accesses those object details using "saving_dir_path" and generates scene graph captions by using "tempura_predcls" model. The relationships detected between human-object pairs for each frame are saved in "output_dir" path.

1. Within scene_graph folder create "data" folder and download glove file there.
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


7. Run the main file:
```python
python final_obj_rel_copy.py -mode predcls -datasize large -data_path /home/eegrad/rghosal/STTran/data/ag/  -model_path /data/AmitRoyChowdhury/Rinki/tempura_models/predcls/best_Mrecall_model.tar -input_dir /data/AmitRoyChowdhury/Rinki/Activity_box_test -original_video_path /data/AmitRoyChowdhury/sayak/activity-net-captions/test_paths_sample.txt -modified_file_path /data/AmitRoyChowdhury/sayak/activity-net-captions/test_paths2.txt -output_dir  /data/AmitRoyChowdhury/Rinki/Activity_test_relation2 -saving_dir_path /data/AmitRoyChowdhury/Rinki/Activity_box_test2 -rel_mem_compute joint -rel_mem_weight_type simple -mem_fusion late -mem_feat_selection manual  -mem_feat_lambda 0.5  -rel_head gmm -obj_head linear -K 6
```
