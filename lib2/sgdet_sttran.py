import numpy as np
import numpy
import torch
import torch.nn as nn

from lib.word_vectors import obj_edge_vectors
from lib.transformer import transformer
from lib.fpn.box_utils import center_size
from fasterRCNN.lib.model.roi_layers import ROIAlign, nms
from lib.draw_rectangles.draw_rectangles import draw_union_boxes
from dataloader.coco import COCO
from lib.config import Config
conf=Config()


class ROI_Feature(nn.Module):
    """
    Module for computing the object contexts and edge contexts
    """

    def __init__(self, mode='sgdet', obj_classes=None):
        super(ROI_Feature, self).__init__()
        self.classes = obj_classes
        self.mode = mode
        print("no of obj class sent while calling ROI_feature",len(self.classes))
        #----------add nms when sgdet
        #self.nms_filter_duplicates = True
        #self.max_per_img =64
        #self.thresh = 0.01

        #roi align
        self.RCNN_roi_align = ROIAlign((7, 7), 1.0/16.0, 0)

        embed_vecs = obj_edge_vectors(obj_classes[1:], wv_type='glove.6B', wv_dir='~/STTran/data/', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes)-1, 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        # This probably doesn't help it much
        self.pos_embed = nn.Sequential(nn.BatchNorm1d(4, momentum=0.01 / 10.0),
                                       nn.Linear(4, 128),
                                       nn.ReLU(inplace=True),
                                       nn.Dropout(0.1))
        self.obj_dim = 2048
        self.decoder_lin = nn.Sequential(nn.Linear(self.obj_dim + 200 + 128, 1024),
                                         nn.BatchNorm1d(1024),
                                         nn.ReLU(),
                                         nn.Linear(1024, len(self.classes)))

    def forward(self, entry):

        box_idx = entry['boxes'][:, 0].long()
        u_f = np.unique(box_idx.cpu().numpy().tolist())    #int(box_idx[-1] + 1)  #max_no_of_frame   ## change into uniqu frames only
        b=len(u_f)

        #print(entry['distribution'].shape)
        obj_embed = entry['distribution'] @ self.obj_embed.weight
        pos_embed = self.pos_embed(center_size(entry['boxes'][:, 1:]))

        print("feature shape:",entry["features"].shape)
        print("obj_embed shape:",obj_embed.shape)
        
        obj_features = torch.cat((entry['features'], obj_embed, pos_embed), 1) #use the result from FasterRCNN directly
        # # use the infered object labels for new pair idx
        HUMAN_IDX = torch.zeros([b, 1], dtype=torch.int64).to(box_idx.device) #??????????????????????????????????
        global_idx = torch.arange(0, entry['boxes'].shape[0])

        for i in range(b):
                #print(entry['distribution'].shape)
                local_human_idx = torch.argmax(entry['distribution'][
                                                        box_idx == i, 0])  # the local bbox index with highest human score in this frame
                HUMAN_IDX[i] = global_idx[box_idx == i][local_human_idx]

        entry['pred_labels'][HUMAN_IDX.squeeze()] = 0 #????????????????????????????????????????????????????????
        entry['pred_scores'][HUMAN_IDX.squeeze()] = entry['distribution'][HUMAN_IDX.squeeze(), 0]

        im_idx = []  # which frame are the relations belong to
        pair = []
        for j, i in enumerate(HUMAN_IDX):
            for m in global_idx[box_idx == j][entry['pred_labels'][box_idx == j] != 0]:  # ??????????????????????????????
                im_idx.append(j)
                pair.append([int(i), int(m)])

        pair = torch.tensor(pair).to(box_idx.device) #????????????????????????????????????????????????
        im_idx = torch.tensor(im_idx, dtype=torch.float).to(box_idx.device)
        entry['pair_idx'] = pair
        entry['im_idx'] = im_idx
        entry['human_idx'] = HUMAN_IDX
        entry['boxes'][:, 1:] = entry['boxes'][:, 1:] * entry['im_info']
        union_boxes = torch.cat((im_idx[:, None], torch.min(entry['boxes'][:, 1:3][pair[:, 0]], entry['boxes'][:, 1:3][pair[:, 1]]),
        torch.max(entry['boxes'][:, 3:5][pair[:, 0]], entry['boxes'][:, 3:5][pair[:, 1]])), 1)

        union_feat = self.RCNN_roi_align(entry['fmaps'], union_boxes)       #??????????????????? how to get "fmaps", self.roi instantiation?
        entry['boxes'][:, 1:] = entry['boxes'][:, 1:] / entry['im_info']
        entry['union_feat'] = union_feat
        entry['union_box'] = union_boxes
        pair_rois = torch.cat((entry['boxes'][pair[:, 0], 1:], entry['boxes'][pair[:, 1], 1:]),
                                       1).data.cpu().numpy()
        entry['spatial_masks'] = torch.tensor(draw_union_boxes(pair_rois, 27) - 0.5).to(box_idx.device) #???????????????????????????
        return entry



class Final_Relation_Feature(nn.Module):

    def __init__(self, mode='sgdet',
                 attention_class_num=None, spatial_class_num=None, contact_class_num=None, obj_classes=None, rel_classes=None,
                 enc_layer_num=None, dec_layer_num=None):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        """
        super(Final_Relation_Feature, self).__init__()
        self.obj_classes = obj_classes
        self.rel_classes = rel_classes
        self.attention_class_num = attention_class_num
        self.spatial_class_num = spatial_class_num
        self.contact_class_num = contact_class_num
        assert mode in ('sgdet', 'sgcls', 'predcls')
        self.mode = mode

        self.union_feature_maker = ROI_Feature(mode=self.mode, obj_classes=self.obj_classes) ######### important

        ###################################
        self.union_func1 = nn.Conv2d(1024, 256, 1, 1)
        self.conv = nn.Sequential(
             nn.Conv2d(2, 256 //2, kernel_size=7, stride=2, padding=3, bias=True),
             nn.ReLU(inplace=True),
             nn.BatchNorm2d(256//2, momentum=0.01),
             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
             nn.Conv2d(256 // 2, 256, kernel_size=3, stride=1, padding=1, bias=True),
             nn.ReLU(inplace=True),
             nn.BatchNorm2d(256, momentum=0.01),
         )
        self.subj_fc = nn.Linear(2048, 512)
        self.obj_fc = nn.Linear(2048, 512)
        self.vr_fc = nn.Linear(256*7*7, 512)
        print("length within relation",len(obj_classes))
        embed_vecs = obj_edge_vectors(obj_classes, wv_type='glove.6B', wv_dir='~/STTran/data/', wv_dim=200)
        self.obj_embed = nn.Embedding(len(obj_classes), 200)
        self.obj_embed.weight.data = embed_vecs.clone()

        self.obj_embed2 = nn.Embedding(len(obj_classes), 200)
        self.obj_embed2.weight.data = embed_vecs.clone()

        self.glocal_transformer = transformer(enc_layer_num=enc_layer_num, dec_layer_num=dec_layer_num, embed_dim=1936, nhead=8,
                                               dim_feedforward=2048, dropout=0.1, mode='latter')

        self.a_rel_compress = nn.Linear(1936, self.attention_class_num)
        self.s_rel_compress = nn.Linear(1936, self.spatial_class_num)
        self.c_rel_compress = nn.Linear(1936, self.contact_class_num)

    def forward(self, entry_saved):

        entry = self.union_feature_maker(entry_saved)

        
        # visual part
        subj_rep = entry['features'][entry['pair_idx'][:, 0]]
        subj_rep = self.subj_fc(subj_rep)
        obj_rep = entry['features'][entry['pair_idx'][:, 1]]
        obj_rep = self.obj_fc(obj_rep)
        vr = self.union_func1(entry['union_feat'])+self.conv(entry['spatial_masks'])
        vr = self.vr_fc(vr.view(-1,256*7*7))
        x_visual = torch.cat((subj_rep, obj_rep, vr), 1)

        # semantic part
        subj_class = entry['pred_labels'][entry['pair_idx'][:, 0]]
        obj_class = entry['pred_labels'][entry['pair_idx'][:, 1]]
        subj_emb = self.obj_embed(subj_class) #############################error
        obj_emb = self.obj_embed2(obj_class)
        x_semantic = torch.cat((subj_emb, obj_emb), 1)

        rel_features = torch.cat((x_visual, x_semantic), dim=1)


        ########### do we need this part?????????????????????????????????????????????????????????
        # Spatial-Temporal Transformer
        # global_output, global_attention_weights, local_attention_weights = self.glocal_transformer(features=rel_features, im_idx=entry['im_idx'])

        # entry["attention_distribution"] = self.a_rel_compress(global_output)
        # entry["spatial_distribution"] = self.s_rel_compress(global_output)
        # entry["contacting_distribution"] = self.c_rel_compress(global_output)

        # entry["spatial_distribution"] = torch.sigmoid(entry["spatial_distribution"])
        # entry["contacting_distribution"] = torch.sigmoid(entry["contacting_distribution"]) 

        return entry                