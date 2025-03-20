# 这里对backbone的所有参数都进行更新，包括backbone的bn层
# 增加正则化操作，防止快速过拟合
import copy
import sys

import math
import torch
from einops import rearrange
from torch import nn
from torch._C import device
import torch.nn.functional as F
from torch.nn import BatchNorm2d as BatchNorm        

import numpy as np
import random
import time
import cv2

from util.util import get_train_val_set
from torch.hub import download_url_to_file
from dropblock import DropBlock2D
from constants import pretrained_weights, model_urls
from pathlib import Path
from . import vit
from model.BDC_module import *
import seaborn as sns
import matplotlib.pyplot as plt

class Residual(nn.Module):
    def __init__(self, layers, up=2):
        super().__init__()
        self.layers = layers
        self.up = up

    def forward(self, x):
        h, w = x.shape[-2:]
        x_up = F.interpolate(x, (h * self.up, w * self.up), mode='bilinear', align_corners=True)
        x = x_up + self.layers(x)
        return x

def get_vgg16_layer(model):
    layer0_idx = range(0,7)
    layer1_idx = range(7,14)
    layer2_idx = range(14,24)
    layer3_idx = range(24,34)
    layer4_idx = range(34,43)
    layers_0 = []
    layers_1 = []
    layers_2 = []
    layers_3 = []
    layers_4 = []
    for idx in layer0_idx:
        layers_0 += [model.features[idx]]
    for idx in layer1_idx:
        layers_1 += [model.features[idx]]
    for idx in layer2_idx:
        layers_2 += [model.features[idx]]
    for idx in layer3_idx:
        layers_3 += [model.features[idx]]
    for idx in layer4_idx:
        layers_4 += [model.features[idx]]  
    layer0 = nn.Sequential(*layers_0) 
    layer1 = nn.Sequential(*layers_1) 
    layer2 = nn.Sequential(*layers_2) 
    layer3 = nn.Sequential(*layers_3) 
    layer4 = nn.Sequential(*layers_4)
    return layer0,layer1,layer2,layer3,layer4

def Weighted_GAP(supp_feat, mask):
    supp_feat = supp_feat * mask
    feat_h, feat_w = supp_feat.shape[-2:][0], supp_feat.shape[-2:][1]
    area = F.avg_pool2d(mask, (supp_feat.size()[2], supp_feat.size()[3])) * feat_h * feat_w + 0.0005
    supp_feat = F.avg_pool2d(input=supp_feat, kernel_size=supp_feat.shape[-2:]) * feat_h * feat_w / area
    return supp_feat

def get_gram_matrix(fea):
    b, c, h, w = fea.shape        
    fea = fea.reshape(b, c, h*w)    # C*N
    fea_T = fea.permute(0, 2, 1)    # N*C
    fea_norm = fea.norm(2, 2, True)
    fea_T_norm = fea_T.norm(2, 1, True)
    gram = torch.bmm(fea, fea_T)/(torch.bmm(fea_norm, fea_T_norm) + 1e-7)    # C*C
    return gram


class OneModel(nn.Module):
    def __init__(self, args, cls_type=None):
        super(OneModel, self).__init__()

        self.cls_type = cls_type  # 'Base' or 'Novel'
        self.zoom_factor = args.zoom_factor
        self.shot = args.shot
        self.dataset = args.data_set
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_label)
        self.print_freq = args.print_freq/2
        self.drop_dim = 1
        self.drop_rate = 0.1
        self.drop2d_kwargs = {'drop_prob': 0.1, 'block_size': 4}
        embed_dim = vit.vit_factory[args.backbone]['embed_dim']

        self.pretrained = True
        self.classes = 2
        if self.dataset == 'pascal':
            self.base_classes = 15
        elif self.dataset == 'coco':
            self.base_classes = 60

        pretrained_vit = self.get_or_download_pretrained(args.backbone, True)
        self.original_encoder = vit.vit_model(args.backbone,
                                            args.train_h,
                                            pretrained=pretrained_vit,
                                            num_classes=0,
                                            opt=args,
                                            original=True)

        self.purifier = self.build_upsampler(embed_dim)
        self.embed_dim = vit.vit_factory[args.backbone]['embed_dim']

        self.bg_sampler = np.random.RandomState(1289)
        self.bdc = BDC(is_vec=True, input_dim=[self.embed_dim,30,30], dimension_reduction=self.embed_dim, activate='relu')
        self.bdc_hw = BDC(is_vec=True, input_dim=[self.embed_dim,30,30], dimension_reduction=self.embed_dim, activate='relu')

        reduce_dim = 256
        self.low_fea_id = args.low_fea[-1]

        self.norm = nn.LayerNorm([self.embed_dim, 30, 30])

        self.mid_features_layer = [4,7]
        self.GAP = nn.AdaptiveAvgPool2d(1)

    def get_optim(self, model, args, LR):

        optimizer = torch.optim.SGD(
            [
            {'params': model.original_encoder.parameters(), 'lr': LR},
            {'params': model.purifier.parameters()},
            {'params': model.bdc.parameters()},
            {'params': model.bdc_hw.parameters()},
            {'params': model.norm.parameters(), 'lr': LR},
            ], lr=LR, momentum=args.momentum, weight_decay=args.weight_decay)
        return optimizer

    def forward(self, x, s_x, s_y, y_m, chosen_cls=None, cat_idx=None, padding_mask=None, s_padding_mask=None, epoch=0):
        x_size = x.size()
        # assert (x_size[2]-1) % 8 == 0 and (x_size[3]-1) % 8 == 0
        H = int((x_size[2] - 1) / 8 * self.zoom_factor + 1)
        W = int((x_size[3] - 1) / 8 * self.zoom_factor + 1)
        B, S, C, H, W = s_x.size()



        #   Support Feature     
        supp_feat_list = []
        final_supp_list = []
        mask_list = []
        supp_mask_cls_token_list = []
        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            supp_mask_out = self.original_encoder(s_x[:,i,:,:,:], mid_features_layer=self.mid_features_layer)   # [8, 768, 29, 29]
            supp_mask_cls_token = supp_mask_out['cls_token']
            supp_mask_cls_token_list.append(supp_mask_cls_token)

        supp_mask_cls_token = torch.cat(supp_mask_cls_token_list, 1)    # [B, s, c, h, w]
        supp_mask_cls_prompt = supp_mask_cls_token.mean(1, keepdim=True)  # B,1,c

        for i in range(self.shot):
            mask = (s_y[:,i,:,:] == 1).float().unsqueeze(1)
            mask_list.append(mask)
            # get support feature with cls_token
            supp_out = self.original_encoder(s_x[:,i,:,:,:], mid_features_layer=self.mid_features_layer, prompt_token=supp_mask_cls_prompt)   # [8, 768, 29, 29]
            supp_feat = supp_out['out']
            _, c, h0, w0 = supp_feat.shape
            supp_feat_list.append(supp_feat)

        # get query feature with cls_token
        query_out = self.original_encoder(x, mid_features_layer=self.mid_features_layer, prompt_token=supp_mask_cls_prompt)    # [8, 768, 29, 29]
        query_feat = query_out['out']

        # bdc enhancement
        query_final_feat, selected_query_feat, selected_query_feat_hw = self.bdc_metric_enhance(supp_feat, query_feat, mask)

        query_final_feat = self.norm(query_final_feat)

        # upsampling
        supp_feat = self.purifier(supp_feat)
        query_feat = self.purifier(query_final_feat)

        sup_fts = supp_feat.unsqueeze(1)    # [B, s, c, h, w]
        qry_fts = query_feat.unsqueeze(1)       # [B, 1, c, h, w]

        sup_mask = F.interpolate(mask, size=(supp_feat.size(-2), supp_feat.size(-1)), mode='bilinear', align_corners=True)  # [B, s, h, w]
        out = self.classifier(sup_fts, qry_fts, sup_mask)          # [B, 2, h, w]
        # support pred
        query_mask = out.max(1)[1].unsqueeze(1)
        out_s = self.classifier(qry_fts, sup_fts, query_mask)          # [B, 2, h, w]

        #   Output Part
        if self.zoom_factor != 1:
            out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=True)
            out_s = F.interpolate(out_s, size=(H, W), mode='bilinear', align_corners=True)

        if self.training:
            main_loss = self.criterion(out, y_m.long())

            aux_loss = self.criterion(out_s, mask.squeeze(1).long()) #+ cls_constrain+0.1

            return out.max(1)[1], main_loss, aux_loss
        else:
            return out

    def build_upsampler(self, embed_dim):
        return Residual(nn.Sequential(
            nn.Conv2d(embed_dim, 256, kernel_size=1),
            nn.ReLU(),
            nn.Dropout(self.drop_rate) if self.drop_dim == 1 else DropBlock2D(**self.drop2d_kwargs),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Dropout(self.drop_rate) if self.drop_dim == 1 else DropBlock2D(**self.drop2d_kwargs),
            nn.Conv2d(256, embed_dim, kernel_size=1),
            nn.ReLU(inplace=True),
        ))

    def bdc_metric_enhance(self, supp_feat, query_feat, mask):
        if len(query_feat.shape) == 3:
            bs, hw, c = query_feat.shape
            h, w = int(math.sqrt(hw)), int(math.sqrt(hw))
            mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=True)
            supp_feat = rearrange(supp_feat, 'b (h w) c -> b c h w', h=h, w=w)
            query_feat = rearrange(query_feat, 'b (h w) c -> b c h w', h=h, w=w)
        else:
            bs, c, h, w = query_feat.shape
            hw = h * w
            mask = F.interpolate(mask, size=(h, w), mode='bilinear', align_corners=True)

        supp_fg_feat = supp_feat.clone() * mask
        supp_bg_feat = supp_feat.clone() * (1 - mask)

        query_feat_c = query_feat.clone().view(query_feat.size(0), query_feat.size(1), -1)
        # bdc eliminate hw
        supp_c_bdc = self.bdc(supp_fg_feat)  # b, c*(c+1)/2
        query_c_bdc = self.bdc(query_feat.clone())
        supp_c_bg_bdc = self.bdc(supp_bg_feat)

        fg_score_matrix = query_c_bdc.transpose(1, 2) @ supp_c_bdc
        bg_score_matrix = query_c_bdc.transpose(1, 2) @ supp_c_bg_bdc

        fg_score_matrix = fg_score_matrix.sum(-1)
        bg_score_matrix = bg_score_matrix.sum(-1)

        c_score_matrix = torch.cat([bg_score_matrix.unsqueeze(1), fg_score_matrix.unsqueeze(1)], 1)
        selected_channel = torch.max(c_score_matrix, dim=1)[1]

        # bdc eliminate channel
        supp_fg_feat_hwc = supp_feat.clone() * mask
        supp_bg_feat_hwc = supp_feat.clone() * (1 - mask)
        query_feat_hwc = query_feat.clone()
        supp_hw_bdc = self.bdc_hw.forward_hwc(supp_fg_feat_hwc)  # b, c*(c+1)/2
        query_hw_bdc = self.bdc_hw.forward_hwc(query_feat_hwc)
        supp_hw_bg_bdc = self.bdc_hw.forward_hwc(supp_bg_feat_hwc)

        query_feat_hwc = rearrange(query_feat_hwc, 'b c h w -> b (h w) c')

        hw_fg_score_matrix = query_hw_bdc.transpose(1, 2) @ supp_hw_bdc
        hw_bg_score_matrix = query_hw_bdc.transpose(1, 2) @ supp_hw_bg_bdc

        hw_fg_score_matrix = hw_fg_score_matrix.sum(-1)
        hw_bg_score_matrix = hw_bg_score_matrix.sum(-1)

        hw_score_matrix = torch.cat([hw_bg_score_matrix.unsqueeze(1), hw_fg_score_matrix.unsqueeze(1)], 1)
        selected_hw = torch.max(hw_score_matrix, dim=1)[1]
        # get the corresponding channel from query feat and keep dimension
        # query feat enhancement by matrix multiplication
        selected_query_feat = []
        selected_query_feat_hw = []
        for i in range(bs):
            if selected_channel[i].sum() == 0:
                tmp_selected_channel = torch.topk(bg_score_matrix[i] - fg_score_matrix[i], k=int(0.1 * c), dim=0)[1]
                tmp_query_feat = query_feat_c[i, tmp_selected_channel[:], :]
            else:
                tmp_query_feat = query_feat_c[i, selected_channel[i, :] == 1, :]
            if selected_hw[i].sum() == 0:
                tmp_selected_hw = torch.topk(hw_bg_score_matrix[i] - hw_fg_score_matrix[i], k=int(0.1 * hw), dim=0)[1]
                tmp_query_feat_hw = query_feat_hwc[i, tmp_selected_hw[:], :]
            else:
                tmp_query_feat_hw = query_feat_hwc[i, selected_hw[i, :] == 1, :]

            query_self_sim = query_feat_c[i, ...] @ tmp_query_feat.T
            enhanced_query_feat = query_self_sim @ tmp_query_feat
            selected_query_feat.append(enhanced_query_feat.unsqueeze(0))
            query_hw_self_sim = query_feat_hwc[i, ...] @ tmp_query_feat_hw.T
            enhanced_query_feat_hw = query_hw_self_sim @ tmp_query_feat_hw
            selected_query_feat_hw.append(enhanced_query_feat_hw.unsqueeze(0))

        selected_query_feat = torch.cat(selected_query_feat, dim=0)
        selected_query_feat_hw = torch.cat(selected_query_feat_hw, dim=0)

        selected_query_feat = rearrange(selected_query_feat, 'b c (h w) -> b c h w', h=h, w=w)
        selected_query_feat_hw = rearrange(selected_query_feat_hw, 'b (h w) c-> b c h w', h=h, w=w)

        query_feat = query_feat + selected_query_feat + selected_query_feat_hw

        return query_feat, selected_query_feat, selected_query_feat_hw

    def classifier(self, sup_fts, qry_fts, sup_mask):
        """

        Parameters
        ----------
        sup_fts: torch.Tensor
            [B, S, c, h, w]
        qry_fts: torch.Tensor
            [B, 1, c, h, w]
        sup_mask: torch.Tensor
            [BS, 1, h, w]

        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w]

        """
        B, S, c, h, w = sup_fts.shape

        # FG proxies
        sup_fg = (sup_mask == 1).view(-1, 1, h * w)  # [BS, 1, hw]
        fg_vecs = torch.sum(sup_fts.reshape(-1, c, h * w) * sup_fg, dim=-1) / (sup_fg.sum(dim=-1) + 1e-5)     # [BS, c]
        # Merge multiple shots
        fg_proto = fg_vecs.view(B, S, c).mean(dim=1)    # [B, c]

        # BG proxies
        bg_proto = self.compute_multiple_prototypes(5, sup_fts, sup_mask == 0, self.bg_sampler)

        # Calculate cosine similarity
        qry_fts = qry_fts.reshape(-1, c, h, w)
        pred = self.compute_similarity(fg_proto, bg_proto, qry_fts)   # [B, 2, h, w]
        return pred

    @staticmethod
    def compute_multiple_prototypes(bg_num, sup_fts, sup_bg, sampler):
        """

        Parameters
        ----------
        bg_num: int
            Background partition numbers
        sup_fts: torch.Tensor
            [B, S, c, h, w], float32
        sup_bg: torch.Tensor
            [BS, 1, h, w], bool
        sampler: np.random.RandomState

        Returns
        -------
        bg_proto: torch.Tensor
            [B, c, k], where k is the number of background proxies

        """
        B, S, c, h, w = sup_fts.shape
        bg_mask = sup_bg.view(B, S, h, w)    # [B, S, h, w]
        batch_bg_protos = []

        for b in range(B):
            bg_protos = []
            for s in range(S):
                bg_mask_i = bg_mask[b, s]     # [h, w]

                # Check if zero
                with torch.no_grad():
                    if bg_mask_i.sum() < bg_num:
                        bg_mask_i = bg_mask[b, s].clone()    # don't change original mask
                        bg_mask_i.view(-1)[:bg_num] = True

                # Iteratively select farthest points as centers of background local regions
                all_centers = []
                first = True
                pts = torch.stack(torch.where(bg_mask_i), dim=1)     # [N, 2]
                for _ in range(bg_num):
                    if first:
                        i = sampler.choice(pts.shape[0])
                        first = False
                    else:
                        dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
                        # choose the farthest point
                        i = torch.argmax((dist ** 2).sum(-1).min(1)[0])
                    pt = pts[i]   # center y, x
                    all_centers.append(pt)
            
                # Assign bg labels for bg pixels
                dist = pts.reshape(-1, 1, 2) - torch.stack(all_centers, dim=0).reshape(1, -1, 2)
                bg_labels = torch.argmin((dist ** 2).sum(-1), dim=1)

                # Compute bg prototypes
                bg_feats = sup_fts[b, s].permute(1, 2, 0)[bg_mask_i]    # [N, c]
                for i in range(bg_num):
                    proto = bg_feats[bg_labels == i].mean(0)    # [c]
                    bg_protos.append(proto)

            bg_protos = torch.stack(bg_protos, dim=1)   # [c, k]
            batch_bg_protos.append(bg_protos)
        bg_proto = torch.stack(batch_bg_protos, dim=0)  # [B, c, k]
        return bg_proto

    @staticmethod
    def compute_similarity(fg_proto, bg_proto, qry_fts, dist_scalar=20):
        """
        Parameters
        ----------
        fg_proto: torch.Tensor
            [B, c], foreground prototype
        bg_proto: torch.Tensor
            [B, c, k], multiple background prototypes
        qry_fts: torch.Tensor
            [B, c, h, w], query features
        dist_scalar: int
            scale factor on the results of cosine similarity

        Returns
        -------
        pred: torch.Tensor
            [B, 2, h, w], predictions
        """
        fg_distance = F.cosine_similarity(
            qry_fts, fg_proto[..., None, None], dim=1) * dist_scalar        # [B, h, w]
        if len(bg_proto.shape) == 3:    # multiple background protos
            bg_distances = []
            for i in range(bg_proto.shape[-1]):
                bg_p = bg_proto[:, :, i]
                bg_d = F.cosine_similarity(
                    qry_fts, bg_p[..., None, None], dim=1) * dist_scalar        # [B, h, w]
                bg_distances.append(bg_d)
            bg_distance = torch.stack(bg_distances, dim=0).max(0)[0]
        else:   # single background proto
            bg_distance = F.cosine_similarity(
                qry_fts, bg_proto[..., None, None], dim=1) * dist_scalar        # [B, h, w]
        pred = torch.stack((bg_distance, fg_distance), dim=1)               # [B, 2, h, w]

        return pred
    
    @staticmethod
    def get_or_download_pretrained(backbone, progress):
        if backbone not in pretrained_weights:
            raise ValueError(f'Not supported backbone {backbone}. '
                             f'Available backbones: {list(pretrained_weights.keys())}')

        cached_file = Path(pretrained_weights[backbone])
        if cached_file.exists():
            return cached_file

        # Try to download
        url = model_urls[backbone]
        cached_file.parent.mkdir(parents=True, exist_ok=True)
        sys.stderr.write('Downloading: "{}" to {}\n'.format(url, cached_file))
        download_url_to_file(url, str(cached_file), progress=progress)
        return cached_file

    def similarity_func(self, feature_q, fg_proto, bg_proto):
        similarity_fg = F.cosine_similarity(feature_q, fg_proto, dim=1)
        similarity_bg = F.cosine_similarity(feature_q, bg_proto, dim=1)

        out = torch.cat((similarity_bg[:, None, ...], similarity_fg[:, None, ...]), dim=1) * 10.0
        return out