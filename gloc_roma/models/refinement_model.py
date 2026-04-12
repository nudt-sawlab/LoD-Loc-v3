import torch
from torch import nn
import numpy as np

from gloc.models.layers import L2Norm
import torch.nn.functional as F


class DenseFeaturesRefiner(nn.Module):
    def __init__(self, conf, ref_model):
        super().__init__() 
        self.conf = conf       
        self.ref_model = ref_model
        self.clamp  = conf.clamp
        self.norm = L2Norm()
        self.conf.bs = 32
        
    def forward(self, x):
        """
        Args:
            x (torch.tensor): batch of images shape Bx3xHxW
        Returns:
            torch.tensor: Features of shape BxDxHxW
        """
        with torch.no_grad():
            desc = self.ref_model(x)

            # desc = self.norm(desc)

        return desc
    
    def score_candidates(self, q_feats, r_db_descriptors):
        """_summary_

        Args:
            q_feats (np.array): shape 1 x C x H x W
            r_db_descriptors (np.array): shape N_cand x C x H x W

        Returns:
            torch.tensor : vector of shape (N_cand, ), score of each one
        """
        q_feats = torch.tensor(q_feats)

        # this version is faster than looped, but requires much more memory due to broadcasting
        # r_db = torch.tensor(r_db_descriptors).squeeze(1)
        # scores = torch.linalg.norm(q_feats - r_db, dim=1) 
        scores = torch.zeros(len(r_db_descriptors), q_feats.shape[-2], q_feats.shape[-1])
        for i, desc in enumerate(r_db_descriptors):
            # q_feats : 1, D, H, W
            # desc    :    D, H, W
            # score   : 1, H, W
            score = torch.linalg.norm(q_feats - torch.tensor(desc), dim=1)
            scores[i] = score[0]

        if self.clamp > 0:
            scores = scores.clamp(max=self.clamp)
        scores = scores.sum(dim=(1,2)) / np.prod(scores.shape[-2:])

        return scores
    
    def find_most_similar_batch(self, rgb_features, target_features):
        """_summary_

        Args:
            q_feats (np.array): shape 1 x C x H x W
            r_db_descriptors (np.array): shape N_cand x C x H x W

        Returns:
            torch.tensor : vector of shape (N_cand, ), score of each one
        """
        """
        找出每个RGB特征最相似的特征的batch序号
        rgb_features: [8, H, W]
        target_features: [72, H, W]
        返回: [8] (每个元素为最相似的特征的batch序号)
        """
        
        rgb_features = torch.tensor(rgb_features)
        target_features = torch.tensor(target_features)
        batch_size_rgb,_, H, W = rgb_features.shape
        batch_size_target,_, _, _ = target_features.shape

        # 归一化特征
        rgb_features_norm = F.normalize(rgb_features.view(batch_size_rgb, -1), p=2, dim=-1)
        target_features_norm = F.normalize(target_features.view(batch_size_target, -1), p=2, dim=-1)

        # 计算相似性矩阵
        similarity_matrix = torch.matmul(rgb_features_norm, target_features_norm.t())  # [8, 72]
        
        # 找出最相似的特征的batch序号
        most_similar_batch_indices = torch.argmax(similarity_matrix, dim=1)  # [8]

        return similarity_matrix.squeeze()# , most_similar_batch_indices
    
    def calc_euclidean_distance(self, F, M):
        """
        计算特征图 F 和模板图集合 M 的欧氏距离，并对距离进行排序。

        参数:
        F (torch.Tensor): 特征图，形状为 (1, 1377, 768)
        M (torch.Tensor): 模板图集合，形状为 (18, 1377, 768)

        返回:
        sorted_distances (list): 每个模板与特征图 F 的欧氏距离，按升序排列
        """
        # 确保输入形状正确
        # breakpoint()
        assert F.shape[1:] == M.shape[1:], "特征图 F 的通道、高度和宽度必须与模板图集合 M 的一致"
        # 确保输入是 PyTorch 张量
        if isinstance(F, np.ndarray):
            F = torch.from_numpy(F)
        if isinstance(M, np.ndarray):
            M = torch.from_numpy(M)

        # 归一化特征
        F = F / torch.norm(F, p=2, dim=1, keepdim=True)
        M = M / torch.norm(M, p=2, dim=1, keepdim=True)
        
        # 计算欧氏距离
        distances = torch.norm(F - M, p=2, dim=(1, 2))  # (18)
        
        # 对距离进行排序（按升序）
        sorted_distances, indices = torch.sort(distances)
        
        return distances
    
    def find_most_IoU_batch(self, mask_feature, target_features):
        """
        找出每个 mask 特征最相似的特征的 batch 序号（基于 IoU）

        Args:
            mask_feature (torch.tensor): shape (1, H, W)
            target_features (torch.tensor): shape (N, H, W)

        Returns:
            torch.tensor: shape (N,), 每个元素表示与 mask_feature 的 IoU
        """

        # 确保输入是 PyTorch 张量
        # mask_feature = torch.tensor(mask_feature, dtype=torch.float32).cuda()
        mask_feature = mask_feature.cuda()
        target_features = torch.tensor(target_features, dtype=torch.float32).cuda()
        
        # 计算交集
        intersection = torch.sum(mask_feature * target_features, dim=(1, 2))

        # 计算并集
        union = torch.sum(mask_feature, dim=(1, 2)) + torch.sum(target_features, dim=(1, 2)) - intersection

        # 计算 IoU
        iou_scores = intersection / union  # [N]

        # 找出最相似的特征的 batch 序号
        most_similar_batch_indices = torch.argmax(iou_scores)  # [1]
        return iou_scores.cpu().squeeze()

    def rank_candidates(self, q_feats, r_db_descriptors, get_scores=False):
        # scores = self.find_most_similar_batch(q_feats, r_db_descriptors)
        # scores = self.calc_euclidean_distance(q_feats, r_db_descriptors)
        scores = self.score_candidates(q_feats, r_db_descriptors)
        preds = torch.argsort(scores)#, descending=True)
        if get_scores:
            return preds, scores[preds]
        return preds
    
    def rank_candidates_mask(self, q_feats, r_db_descriptors, get_scores=False):
        scores = self.find_most_IoU_batch(q_feats.unsqueeze(0), r_db_descriptors)
        preds = torch.argsort(-scores)#, descending=True)
        if get_scores:
            return preds, scores[preds]
        return preds
