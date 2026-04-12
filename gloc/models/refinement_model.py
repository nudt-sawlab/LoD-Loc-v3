import torch
from torch import nn
import numpy as np
import cv2

from gloc.models.layers import L2Norm
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

class DenseFeaturesRefiner(nn.Module):
    def __init__(self, conf, ref_model):
        super().__init__() 
        self.conf = conf# 配置      
        self.ref_model = ref_model# 引用的模型
        self.clamp  = conf.clamp
        self.norm = L2Norm()# L2归一化
        self.conf.bs = 32
        
    def forward(self, x):
        """
        Args:
            x (torch.tensor): batch of images shape Bx3xHxW
        Returns:
            torch.tensor: Features of shape BxDxHxW
        """
        with torch.no_grad():
            # 输入图像 x，获得特征描述 desc。
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
        # q_feats为查询图，r_db_descriptors为候选描述图
        q_feats = torch.tensor(q_feats)

        # this version is faster than looped, but requires much more memory due to broadcasting
        # r_db = torch.tensor(r_db_descriptors).squeeze(1)
        # scores = torch.linalg.norm(q_feats - r_db, dim=1) 
        # 初始化 scores 张量，形状为 (N_cand, H, W)，用于存储每个候选描述符的评分。
        scores = torch.zeros(len(r_db_descriptors), q_feats.shape[-2], q_feats.shape[-1])
        # 遍历每个候选描述符并计算与查询特征之间的 L2 范数（欧氏距离）。
        # 将计算出的分数存入 scores 张量。
        for i, desc in enumerate(r_db_descriptors):
            # q_feats : 1, D, H, W
            # desc    :    D, H, W
            # score   : 1, H, W
            # 在第一维度上求欧氏距离
            score = torch.linalg.norm(q_feats - torch.tensor(desc), dim=1)
            scores[i] = score[0]

        if self.clamp > 0:
            scores = scores.clamp(max=self.clamp)
        # 对每个候选描述符的得分进行归一化，即计算每个候选在空间维度上的平均分数。
        scores = scores.sum(dim=(1,2)) / np.prod(scores.shape[-2:])

        return scores
    
    def find_most_similar_batch(self, rgb_features, target_features):
        """_summary_

        Args:
            rgb_features就是q_feats
            q_feats (np.array): shape 1 x C x H x W
            target_features就是r_db_descriptors
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
        # 获取 RGB 特征和目标特征的批量大小和尺寸。
        batch_size_rgb,_, H, W = rgb_features.shape
        batch_size_target,_, _, _ = target_features.shape

        # 归一化特征，将 RGB 特征和目标特征在最后一维进行 L2 归一化。
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
        # (Pdb) intersection
        # tensor([77404.1406, 77404.1406, 76803.7031, 78117.1406, 78131.8906, 78610.9688,
        #         75300.7578, 81332.4531, 77470.8594, 72735.8359, 66357.0000, 69243.9062,
        #         70216.5000, 75826.9062, 74974.0000, 76667.3516, 70993.9375, 75582.0156,
        #         78698.2344, 75618.0547, 75012.8906, 72865.2969, 79060.3438, 69175.1484,
        #         75408.2812, 67718.9531], device='cuda:0')
        # 计算并集
        union = torch.sum(mask_feature, dim=(1, 2)) + torch.sum(target_features, dim=(1, 2)) - intersection

        # 计算 IoU
        iou_scores = intersection / union  # [N]
        # (Pdb) iou_scores
        # tensor([0.6681, 0.6681, 0.6598, 0.6841, 0.6719, 0.6771, 0.6367, 0.7105, 0.6700,
        #         0.6016, 0.5460, 0.5689, 0.5941, 0.6493, 0.6439, 0.6719, 0.5816, 0.6426,
        #         0.6761, 0.6576, 0.6492, 0.6217, 0.6869, 0.5561, 0.6488, 0.5485],
        # device='cuda:0')
        # 找出最相似的特征的 batch 序号
        most_similar_batch_indices = torch.argmax(iou_scores)  # [1] 用不上
        return iou_scores.cpu().squeeze()
    
    def find_weighted_IoU_multi_class(self, mask_feature, target_features, class_labels, weights=None):
        """
        计算每个 mask 特征与 target_features 中各类别的加权 IoU。

        Args:
            mask_feature (torch.tensor): shape (1, H, W)，包含多类别（用不同值/颜色编码）。
            target_features (torch.tensor): shape (N, H, W)，每个样本可能包含多个类别。
            class_labels (List[int]): 类别标签值（如 [0, 1, 2]），对应颜色或值。
            weights (List[float], optional): 每个类别的权重。如果为 None，则默认平均权重。

        Returns:
            torch.tensor: shape (N,), 表示每个 target 与 mask 的加权平均 IoU。
        """
        mask_feature = mask_feature.cuda()
        target_features = torch.from_numpy(target_features).cuda()
        N, H, W = target_features.shape
        C = len(class_labels)

        if weights is None:
            weights = [1.0 / C] * C
        else:
            weights = torch.tensor(weights, device=mask_feature.device)
            weights = weights / weights.sum()  # 归一化

        iou_scores = torch.zeros(N, device=mask_feature.device)

        for i, cls in enumerate(class_labels):
            # 生成当前类别的二值掩码
            mask_bin = (mask_feature == cls).float()  # (1, H, W)
            targets_bin = (target_features == cls).float()  # (N, H, W)

            # 计算 IoU
            intersection = torch.sum(mask_bin * targets_bin, dim=(1, 2))# torch.Size([一张图1束光多少张候选图])
            union = torch.sum(mask_bin, dim=(1, 2)) + torch.sum(targets_bin, dim=(1, 2)) - intersection
            iou_per_class = intersection / (union + 1e-6)  # 避免除以 0

            iou_scores += weights[i] * iou_per_class  # 加权

        return iou_scores.cpu()
    
    def find_weighted_IoU_from_RGB(self, mask_feature, target_features, weights=None):
        """
        计算每个 target 与 mask_feature 之间的加权 IoU（支持多类 RGB 掩码）。

        Args:
            mask_feature (torch.Tensor): (1, 3, H, W)，RGB 掩码图像
            target_features (torch.Tensor): (N, 3, H, W)，每个 target 是 RGB 图像
            weights (List[float], optional): 类别权重列表。如果为 None，则均匀加权

        Returns:
            torch.Tensor: (N,), 每个 target 的加权 IoU
        """
        mask_feature = mask_feature#.cuda()
        target_features = target_features#.cuda()  # (N, 3, H, W)
        N, _, H, W = target_features.shape

        # 获取所有类别颜色 (C, 3)，每行为一个 RGB 类别
        unique_colors = torch.unique(target_features.permute(0, 2, 3, 1).reshape(-1, 3), dim=0)
        C = unique_colors.shape[0]

        if weights is None:
            weights = torch.ones(C, device=mask_feature.device) / C
        else:
            weights = torch.tensor(weights, device=mask_feature.device)
            weights = weights / weights.sum()  # 归一化

        iou_scores = torch.zeros(N, device=mask_feature.device)

        for i, color in enumerate(unique_colors):
            # (1, H, W) boolean mask for current color
            mask_bin = torch.all(mask_feature[0] == color[:, None, None], dim=0).float()  # (H, W)
            targets_bin = torch.all(target_features == color[None, :, None, None], dim=1).float()  # (N, H, W)

            # IoU
            intersection = torch.sum(mask_bin * targets_bin, dim=(1, 2))  # (N,)
            union = torch.sum(mask_bin, dim=(0, 1)) + torch.sum(targets_bin, dim=(1, 2)) - intersection
            iou_per_class = intersection / (union + 1e-6)

            iou_scores += weights[i] * iou_per_class

        return iou_scores.cpu()



    def rank_candidates(self, q_feats, r_db_descriptors, get_scores=False):
        scores = self.find_most_similar_batch(q_feats, r_db_descriptors)
        # scores = self.calc_euclidean_distance(q_feats, r_db_descriptors)
        # scores = self.score_candidates(q_feats, r_db_descriptors)
        preds = torch.argsort(-scores)#, descending=True)
        if get_scores:
            return preds, scores[preds]
        return preds
    
    # (Pdb) q_feats
    # tensor([[0.9020, 0.9176, 0.9490,  ..., 0.0000, 0.0000, 0.0000],
    #         [0.9137, 0.9255, 0.9529,  ..., 0.0000, 0.0000, 0.0000],
    #         [0.9294, 0.9373, 0.9608,  ..., 0.0000, 0.0000, 0.0000],
    #         ...,
    #         [0.2980, 0.2902, 0.2667,  ..., 0.0000, 0.0039, 0.0039],
    #         [0.2510, 0.2510, 0.2510,  ..., 0.0039, 0.0039, 0.0039],
    #         [0.2275, 0.2314, 0.2431,  ..., 0.0039, 0.0039, 0.0039]],
    # torch.Size([448, 602])

    def rank_candidates_mask(self, q_feats, r_db_descriptors, get_scores=False):
        scores = self.find_most_IoU_batch(q_feats.unsqueeze(0), r_db_descriptors)
        preds = torch.argsort(-scores)#, descending=True)
        if get_scores:
            return preds, scores[preds]
        return preds

    def rank_candidates_multimask(self, q_feats, r_db_descriptors, get_scores=False):
        class_label = [0,1]
        # breakpoint()
        # unsqueeze(0)就是在前面加一个维度，torch.Size([448, 602])🔜torch.Size([1, 448, 602])，用于批量化？
        scores = self.find_weighted_IoU_multi_class(q_feats.unsqueeze(0), r_db_descriptors, class_label)
        #++++++++++++
        # 用wiou再重新计算scores前10的评分，覆盖scores，再返回preds, scores[preds]
        # 但是现在问题是实例没有输入
        #++++++++++++
        preds = torch.argsort(-scores)#, descending=True)
        if get_scores:
            return preds, scores[preds]
        return preds

    def rank_candidates_multimask_twostages(self, q_feats, r_db_descriptors, q_weights=None, topk=10, get_scores=False):
        """
        两阶段实例分割评分：
        1. 第一阶段：只用前景IoU筛选topk
        2. 第二阶段：用实例概率npy和加权dice loss对topk再排序
        """
        # 第一阶段：只用前景IoU（假设前景类别为1）
        class_label = [1]  # 只用前景
        scores = self.find_weighted_IoU_multi_class(q_feats.unsqueeze(0), r_db_descriptors, class_label)
        topk = min(topk, len(scores))
        topk_indices = torch.argsort(-scores)[:topk]
        # 第二阶段：加载npy，计算加权dice loss
        if q_weights is not None:
            from tloc.refine_diceloss import batch_dice_loss
            # 取topk候选的掩码
            candidate_masks = torch.from_numpy(r_db_descriptors[topk_indices.cpu().numpy()]).float().cuda()  # [topk, H, W]
            # dice loss越小越好，这里取负号作为分数越大越好
            dice_losses = batch_dice_loss(probs, candidate_masks, weights)  # [topk]
            final_scores = -dice_losses
            final_indices = topk_indices[torch.argsort(-final_scores)]
            if get_scores:
                return final_indices, final_scores[torch.argsort(-final_scores)]
            return final_indices
        else:
            if get_scores:
                preds = torch.argsort(-scores)#, descending=True)
                return preds, scores[preds]
            return preds

    def rank_candidates_mask_dice(self, q_probs, q_weights, r_db_descriptors, get_scores=False):
        """
        使用实例分割和Dice Loss进行候选排名
        
        Args:
            q_probs: 查询图的实例概率矩阵 [C, H, W]
            q_weights: 每个实例的权重 [C]
            r_db_descriptors: 渲染图描述符 [N, H, W]
            get_scores: 是否返回分数
        
        Returns:
            preds: 排序后的预测索引
            scores: 对应的分数（如果get_scores=True）
        """
        scores = self.calculate_dice_scores(q_probs, q_weights, r_db_descriptors)
        preds = torch.argsort(-scores)  # 降序排列
        if get_scores:
            return preds, scores[preds]
        return preds
    
    def calculate_dice_scores(self, q_probs, q_weights, r_db_descriptors):
        """
        计算基于Dice Loss的分数
        
        Args:
            q_probs: 查询图的实例概率矩阵 [C, H, W]
            q_weights: 每个实例的权重 [C]
            r_db_descriptors: 渲染图描述符 [N, H, W]
        
        Returns:
            scores: 每个渲染图的分数 [N]
        """
        # 确保输入在GPU上
        q_probs = q_probs.cuda()
        q_weights = q_weights.cuda()
        r_db_descriptors = torch.tensor(r_db_descriptors, dtype=torch.float32).cuda()
        
        C, H, W = q_probs.shape
        N = r_db_descriptors.shape[0]
        
        # 扩展维度进行批量计算
        q_probs_expanded = q_probs.unsqueeze(0).expand(N, C, H, W)  # [N, C, H, W]
        r_db_expanded = r_db_descriptors.unsqueeze(1)  # [N, 1, H, W]
        
        # 计算交集
        intersection = (q_probs_expanded * r_db_expanded).sum(dim=(2, 3))  # [N, C]
        
        # 计算Dice系数
        dice = (2 * intersection + 1e-6) / (
            q_probs_expanded.sum(dim=(2, 3)) + r_db_expanded.sum(dim=(2, 3)) + 1e-6
        )  # [N, C]
        
        # 使用权重计算加权平均分数
        weighted_scores = (dice * q_weights.unsqueeze(0)).sum(dim=1)  # [N]
        
        return weighted_scores.cpu()


    def rank_candidates_mask_dice_bboxs(self, q_probs, q_bboxs, r_db_descriptors, weight_type='uniform', q_weights=None, get_scores=False):
        """
        使用实例分割和候选框位置进行Dice Loss计算的候选排名
        
        Args:
            q_probs: 查询图的实例概率矩阵 [C, H, W]
            q_bboxs: 每个实例的候选框 [C, 4] 格式为 [x1, y1, x2, y2]
            r_db_descriptors: 渲染图描述符 [N, H, W]
            weight_type: 权重计算方式 'uniform'(均匀权重)、'area'(面积权重) 或 'scores'(评分权重)
            q_weights: 自定义权重 [C]，当weight_type='scores'时使用
            get_scores: 是否返回分数
        
        Returns:
            preds: 排序后的预测索引
            scores: 对应的分数（如果get_scores=True）
        """
        # scores = self.calculate_dice_scores_bboxs(q_probs, q_bboxs, r_db_descriptors, weight_type, q_weights)
        scores = self.calculate_dice_scores_bboxs_double(q_probs, q_bboxs, r_db_descriptors, weight_type, q_weights, sampling_ratio=1)
        preds = torch.argsort(-scores)  # 降序排列
        if get_scores:
            return preds, scores[preds]
        return preds

    def calculate_dice_scores_bboxs(self, q_probs, q_bboxs, r_db_descriptors, weight_type='uniform', q_weights=None):
        """
        基于候选框计算Dice Loss分数
        
        Args:
            q_probs: 查询图的实例概率矩阵 [C, H, W]
            q_bboxs: 每个实例的候选框 [C, 4] 格式为 [x1, y1, x2, y2]
            r_db_descriptors: 渲染图描述符 [N, H, W]
            weight_type: 权重计算方式 'uniform'、'area' 或 'scores'
            q_weights: 自定义权重 [C]，当weight_type='scores'时使用
        
        Returns:
            scores: 每个渲染图的分数 [N]
        """
        # 确保输入在GPU上
        q_probs = q_probs.cuda()
        q_bboxs = torch.tensor(q_bboxs, dtype=torch.int32).cuda()
        r_db_descriptors = torch.tensor(r_db_descriptors, dtype=torch.float32).cuda() # rgb的话，是torch.Size([26, 3, 360, 640])
        # breakpoint()
        C, H, W = q_probs.shape
        N = r_db_descriptors.shape[0]# 一束光渲染26张图
        # breakpoint()
        # 计算权重
        if weight_type == 'uniform':
            weights = torch.ones(C, device=q_probs.device) / C
        elif weight_type == 'area':
            # 计算每个候选框的面积
            areas = (q_bboxs[:, 2] - q_bboxs[:, 0]) * (q_bboxs[:, 3] - q_bboxs[:, 1])
            weights = areas.float() / areas.sum().float()
        elif weight_type == 'scores':
            # 使用传入的实例评分作为权重
            if q_weights is None:
                raise ValueError("当weight_type='scores'时，必须提供q_weights参数")
            weights = q_weights.cuda() if not q_weights.is_cuda else q_weights
        else:
            raise ValueError(f"不支持的权重类型: {weight_type}")
        
        # 初始化分数张量
        final_scores = torch.zeros(N, device=q_probs.device)
        
        # 对每个实例计算Dice Loss
        for c in range(C):
            # 获取当前实例的候选框坐标
            x1, y1, x2, y2 = q_bboxs[c]
            
            # 确保坐标在有效范围内
            x1 = max(0, min(x1.item(), W-1))
            y1 = max(0, min(y1.item(), H-1))
            x2 = max(x1+1, min(x2.item(), W))
            y2 = max(y1+1, min(y2.item(), H))
            
            # 提取候选框区域
            q_prob_roi = q_probs[c, y1:y2, x1:x2]  # [roi_h, roi_w]
            # r_db_roi = r_db_descriptors[:, y1:y2, x1:x2]  # [N, roi_h, roi_w]
            # 修正后：
            r_db_roi = r_db_descriptors[:, :, y1:y2, x1:x2]  # [N, 3, roi_h, roi_w] - 正确
            # breakpoint()
            # ========================================================================================
            # 新增：处理多实例彩色掩码
            dice_scores = torch.zeros(N, device=q_probs.device)# 26，一束光渲染26张图
            
            for n in range(N):
                # 获取当前渲染图的ROI区域 [3, roi_h, roi_w]
                current_roi = r_db_roi[n]  # [3, roi_h, roi_w]
                
                # 将RGB转换为二维标签图，将RGB值转换为唯一的颜色ID，将RGB三通道合并为单一标识符
                current_roi_2d = (current_roi[0] * 65536 + current_roi[1] * 256 + current_roi[2]).int()
                
                # 使用cv2进行连通组件分析
                # 将tensor转换为numpy数组
                roi_numpy = current_roi_2d.cpu().numpy().astype(np.uint32)
                
                # 识别不同颜色
                unique_colors = np.unique(roi_numpy)
                unique_colors = unique_colors[unique_colors > 0]  # 排除背景(0)
                
                if len(unique_colors) == 0:
                    dice_scores[n] = 0.0
                    continue
                
                # 为每个颜色/实例计算与query的IOU
                max_iou = 0.0
                best_instance_mask = None
                
                for color in unique_colors:
                    # 创建当前颜色的二值掩码
                    color_mask = (roi_numpy == color).astype(np.uint8)
                    
                    # 使用cv2进行连通组件分析
                    num_labels, labels = cv2.connectedComponents(color_mask, connectivity=8)
                    
                    # 对每个连通组件分别计算IOU
                    for label_id in range(1, num_labels):  # 跳过背景标签0
                        # 创建当前连通组件的掩码
                        component_mask = (labels == label_id).astype(np.float32)
                        
                        # 转换回tensor
                        component_mask_tensor = torch.from_numpy(component_mask).to(q_prob_roi.device)
                        
                        # 计算IOU
                        intersection = (q_prob_roi * component_mask_tensor).sum()
                        union = q_prob_roi.sum() + component_mask_tensor.sum() - intersection
                        iou = intersection / (union + 1e-6)
                        
                        # 保存最佳匹配
                        if iou > max_iou:
                            max_iou = iou
                            best_instance_mask = component_mask_tensor
                    
                # 使用最佳匹配的实例计算Dice系数
                if best_instance_mask is not None:
                    intersection = (q_prob_roi * best_instance_mask).sum()
                    q_sum = q_prob_roi.sum()
                    r_sum = best_instance_mask.sum()
                    dice = (2 * intersection + 1e-6) / (q_sum + r_sum + 1e-6)
                    dice_scores[n] = dice
                else:
                    dice_scores[n] = 0.0
            
            # 加权累加到最终分数
            final_scores += weights[c] * dice_scores
            # ========================================================================================
            # # 计算交集
            # intersection = (q_prob_roi.unsqueeze(0) * r_db_roi).sum(dim=(1, 2))  # [N]
            
            # # 计算Dice系数
            # q_sum = q_prob_roi.sum()
            # r_sum = r_db_roi.sum(dim=(1, 2))  # [N]
            
            # dice = (2 * intersection + 1e-6) / (q_sum + r_sum + 1e-6)  # [N]
            
            # # 加权累加到最终分数
            # final_scores += weights[c] * dice
        
        return final_scores.cpu()
# =======================================================================
    # def compute_instance_diou(self, mask1, mask2):
    #     """
    #     计算两个实例掩码的DIoU (Distance IoU)
    #     Args:
    #         mask1: 二值掩码 [H,W], 取值范围{0,1}
    #         mask2: 二值掩码 [H,W], 取值范围{0,1}
    #     Returns:
    #         diou: 标量值，范围[-1,1]
    #     """
    #     # 计算IoU基础部分
    #     intersection = torch.logical_and(mask1, mask2).sum()
    #     union = torch.logical_or(mask1, mask2).sum()
    #     iou = intersection / (union + 1e-6)

    #     # 计算中心点距离
    #     coords1 = mask1.nonzero()
    #     coords2 = mask2.nonzero()
    #     if coords1.shape[0] == 0 or coords2.shape[0] == 0:
    #         return 0.0  # 任意一方无实例时返回0
        
    #     # 计算最小包围框
    #     min_xy = torch.min(
    #         torch.cat([coords1.min(dim=0)[0], coords2.min(dim=0)[0]]), 
    #         dim=0
    #     )[0]
    #     max_xy = torch.max(
    #         torch.cat([coords1.max(dim=0)[0], coords2.max(dim=0)[0]]),
    #         dim=0
    #     )[0]
        
    #     # 计算对角线长度
    #     c = torch.sum((max_xy - min_xy) ** 2)  # 对角线长度平方
        
    #     # 计算中心点距离
    #     center1 = coords1.float().mean(dim=0)
    #     center2 = coords2.float().mean(dim=0)
    #     d = torch.sum((center1 - center2) ** 2)  # 中心距离平方
        
    #     # DIoU公式
    #     diou = iou - (d / (c + 1e-6))
    #     return diou.item()

    # def compute_instance_giou(self, mask1, mask2):
    #     """
    #     计算两个实例掩码的GIoU (Generalized IoU)
    #     Args:
    #         mask1: 二值掩码 [H,W], 取值范围{0,1}
    #         mask2: 二值掩码 [H,W], 取值范围{0,1}
    #     Returns:
    #         giou: 标量值，范围[-1,1]
    #     """
    #     intersection = torch.logical_and(mask1, mask2).sum()
    #     union = torch.logical_or(mask1, mask2).sum()
        
    #     # 计算最小包围框面积
    #     coords = torch.cat([mask1.nonzero(), mask2.nonzero()])
    #     if coords.shape[0] == 0:  # 无交集且无独立区域
    #         return 0.0
        
    #     min_coord = coords.min(dim=0)[0]
    #     max_coord = coords.max(dim=0)[0]
    #     convex_area = (max_coord[0] - min_coord[0] + 1) * (max_coord[1] - min_coord[1] + 1)
        
    #     # GIoU公式
    #     iou = intersection / (union + 1e-6)
    #     giou = iou - (convex_area - union) / (convex_area + 1e-6)
    #     return giou.item()

    # # def instance_matching(self, query_instances, render_instances):
    # #     """
    # #     实例级匹配（双向最优匹配）
    # #     Args:
    # #         query_instances: 列表，每个元素为(query_mask, area_weight)
    # #         render_instances: 列表，每个元素为(render_mask, _)
    # #     Returns:
    # #         matched_scores: 匹配对得分总和
    # #         un_matched_q: 未匹配的查询实例数
    # #         un_matched_r: 未匹配的渲染实例数
    # #     """
    # #     # 构建代价矩阵
    # #     cost_matrix = np.zeros((len(query_instances), len(render_instances)))
    # #     for i, (q_mask, _) in enumerate(query_instances):
    # #         for j, r_mask in enumerate(render_instances):
    # #             cost_matrix[i,j] = 1 - self.compute_instance_giou(q_mask, r_mask)  # 转换为最小化问题
        
    # #     # 匈牙利算法匹配
    # #     row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
    # #     # 筛选有效匹配（GIoU>阈值）
    # #     matched_pairs = []
    # #     for i, j in zip(row_ind, col_ind):
    # #         giou = 1 - cost_matrix[i,j]
    # #         if giou > 0.3:  # 可调阈值
    # #             matched_pairs.append((i, j, giou))
    # #         # matched_pairs.append((i, j, giou))
        
    # #     # 计算未匹配数量
    # #     matched_q = set(i for i,_,_ in matched_pairs)
    # #     matched_r = set(j for _,j,_ in matched_pairs)
    # #     un_matched_q = len(query_instances) - len(matched_q)
    # #     un_matched_r = len(render_instances) - len(matched_r)
        
    # #     # 加权得分
    # #     total_score = sum(giou * query_instances[i][1] for i,j,giou in matched_pairs)
    # #     return total_score, un_matched_q, un_matched_r
    
    # def instance_matching(self, query_instances, render_instances):
    #     """
    #     改进的实例级匹配（支持预测实例数 < 渲染实例数）
    #     Args:
    #         query_instances: [(q_mask, q_weight), ...] 
    #         render_instances: [r_mask, ...]
    #     """
    #     n_query = len(query_instances)
    #     n_render = len(render_instances)
        
    #     # 动态调整矩阵尺寸
    #     max_dim = max(n_query, n_render)
    #     cost_matrix = np.full((max_dim, max_dim), 1e6)  # 用极大值初始化

    #     # 填充实际匹配代价（1 - GIoU）
    #     for i, (q_mask, _) in enumerate(query_instances):
    #         for j, r_mask in enumerate(render_instances):
    #             giou = self.compute_instance_giou(q_mask, r_mask)
    #             cost_matrix[i, j] = 1 - giou  # 转换为最小化问题
    #             # cost_matrix[i,j] = 1 - self.compute_instance_diou(q_mask, r_mask)  # 这里改为DIoU

    #     # 虚拟节点处理（允许不匹配）
    #     if n_query < max_dim:
    #         cost_matrix[n_query:, :] = 1e6  # 禁止虚拟查询节点匹配真实渲染实例
    #     if n_render < max_dim:
    #         cost_matrix[:, n_render:] = 1e6  # 禁止虚拟渲染节点匹配真实查询实例

    #     # 执行匈牙利算法
    #     row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
    #     # 解析匹配结果
    #     matched_pairs = []
    #     valid_indices = (row_ind < n_query) & (col_ind < n_render)
    #     for i, j in zip(row_ind[valid_indices], col_ind[valid_indices]):
    #         giou = 1 - cost_matrix[i, j]
    #         if giou > 0.3:  # GIoU阈值过滤
    #             matched_pairs.append((i, j, giou))

    #     # 统计未匹配数量
    #     matched_q = set(i for i,_,_ in matched_pairs)
    #     matched_r = set(j for _,j,_ in matched_pairs)
        
    #     return (
    #         sum(giou * query_instances[i][1] for i, _, giou in matched_pairs),
    #         len(query_instances) - len(matched_q),
    #         len(render_instances) - len(matched_r)
    #     )


    # def rank_candidates_instance_giou(self, q_feats, r_db_descriptors, get_scores=False):
    #     """
    #     基于实例级GIoU的候选排序
    #     Args:
    #         q_feats: 查询图的实例分割结果 [H,W] 整型矩阵
    #         r_db_descriptors: 候选渲染图列表，每个元素为[H,W]二值掩码
    #     """
    #     # 提取查询实例
    #     query_instances = self.extract_instances(q_feats)  # [(mask, weight), ...]
        
    #     scores = []
    #     for r_mask in r_db_descriptors:
    #         # 提取渲染实例
    #         render_instances = self.extract_instances(r_mask)
            
    #         # # 实例匹配与得分计算
    #         # matched_score, un_q, un_r = self.instance_matching(query_instances, render_instances)
    #         # 执行自适应匹配
    #         matched_score, un_q, un_r = self.instance_matching(query_instances,render_instances)
    #         # 动态惩罚策略（惩罚系数与实例数量相关）
    #         penalty = (un_q * 0.2) + (un_r * 0.1 * (len(render_instances)/len(query_instances)))
    #         final_score = matched_score - penalty
            
    #         # scores.append(final_score)
    #         # # 未匹配惩罚项
    #         # penalty = (un_q + un_r) * 0.1  # 惩罚系数可调
    #         # final_score = matched_score - penalty
            
    #         scores.append(final_score)
        
    #     scores = torch.tensor(scores)
    #     preds = torch.argsort(-scores)
    #     return (preds, scores[preds]) if get_scores else preds

    # def extract_instances(self, mask):
    #     """
    #     从分割掩码中提取实例及权重
    #     Args:
    #         mask: 整型矩阵，0为背景，其他值为实例ID
    #     Returns:
    #         instances: 列表，元素为(mask_tensor, area_weight)
    #     确保能够正确提取实例掩码！！！！！
    #     """
    #     instances = []
    #     instance_ids = torch.unique(mask)
    #     instance_ids = instance_ids[instance_ids != 0]
        
    #     total_area = sum((mask == iid).sum().item() for iid in instance_ids)
    #     for iid in instance_ids:
    #         ins_mask = (mask == iid)
    #         area = ins_mask.sum().item()
    #         weight = area / (total_area + 1e-6)  # 面积权重归一化
    #         instances.append((ins_mask, weight))
    #     return instances

    def calculate_dice_scores_bboxs_double(self, q_probs, q_bboxs, r_db_descriptors, weight_type='uniform', q_weights=None, sampling_ratio=1.0):
        """
        双向匹配的Dice Loss计算：query->render + render->query
        
        Args:
            q_probs: 查询图的实例概率矩阵 [C, H, W]
            q_bboxs: 每个实例的候选框 [C, 4] 格式为 [x1, y1, x2, y2]
            r_db_descriptors: 渲染图描述符 [N, 3, H, W] (RGB格式)
            weight_type: 权重计算方式 'uniform'、'area' 或 'scores'
            q_weights: 自定义权重 [C]，当weight_type='scores'时使用
            sampling_ratio: render实例的抽样比例 (0.0-1.0)
        
        Returns:
            scores: 每个渲染图的分数 [N]
        """
        # 确保输入在GPU上
        q_probs = q_probs.cuda()
        q_bboxs = torch.tensor(q_bboxs, dtype=torch.int32).cuda()
        r_db_descriptors = torch.tensor(r_db_descriptors, dtype=torch.float32).cuda()
        
        C, H, W = q_probs.shape
        N = r_db_descriptors.shape[0]
        
        # 计算权重
        if weight_type == 'uniform':
            weights = torch.ones(C, device=q_probs.device) / C
        elif weight_type == 'area':
            areas = (q_bboxs[:, 2] - q_bboxs[:, 0]) * (q_bboxs[:, 3] - q_bboxs[:, 1])
            weights = areas.float() / areas.sum().float()
        elif weight_type == 'scores':
            if q_weights is None:
                raise ValueError("当weight_type='scores'时，必须提供q_weights参数")
            weights = q_weights.cuda() if not q_weights.is_cuda else q_weights
            # 对scores进行归一化，确保权重和为1
            weights = weights / weights.sum()
        else:
            raise ValueError(f"不支持的权重类型: {weight_type}")
        
        # 初始化分数张量
        final_scores = torch.zeros(N, device=q_probs.device)
        
        # ==================== 第一部分：Query -> Render 匹配 ====================
        # 对每个query实例计算与render实例的匹配
        for c in range(C):
            x1, y1, x2, y2 = q_bboxs[c]
            
            # 确保坐标在有效范围内
            x1 = max(0, min(x1.item(), W-1))
            y1 = max(0, min(y1.item(), H-1))
            x2 = max(x1+1, min(x2.item(), W))
            y2 = max(y1+1, min(y2.item(), H))
            
            # 提取候选框区域
            q_prob_roi = q_probs[c, y1:y2, x1:x2]  # [roi_h, roi_w]
            r_db_roi = r_db_descriptors[:, :, y1:y2, x1:x2]  # [N, 3, roi_h, roi_w]
            
            dice_scores_q2r = torch.zeros(N, device=q_probs.device)
            
            for n in range(N):
                current_roi = r_db_roi[n]  # [3, roi_h, roi_w]
                
                # RGB转换为颜色ID
                current_roi_2d = (current_roi[0] * 65536 + current_roi[1] * 256 + current_roi[2]).int()
                roi_numpy = current_roi_2d.cpu().numpy().astype(np.uint32)
                
                # 识别不同颜色
                unique_colors = np.unique(roi_numpy)
                unique_colors = unique_colors[unique_colors > 0]
                
                if len(unique_colors) == 0:
                    dice_scores_q2r[n] = 0.0
                    continue
                
                # 找到最佳匹配的render实例
                max_iou = 0.0
                best_instance_mask = None
                
                for color in unique_colors:
                    color_mask = (roi_numpy == color).astype(np.uint8)
                    num_labels, labels = cv2.connectedComponents(color_mask, connectivity=8)
                    
                    for label_id in range(1, num_labels):
                        component_mask = (labels == label_id).astype(np.float32)
                        component_mask_tensor = torch.from_numpy(component_mask).to(q_prob_roi.device)
                        
                        intersection = (q_prob_roi * component_mask_tensor).sum()
                        union = q_prob_roi.sum() + component_mask_tensor.sum() - intersection
                        iou = intersection / (union + 1e-6)
                        
                        if iou > max_iou:
                            max_iou = iou
                            best_instance_mask = component_mask_tensor
                    
                # 计算Dice系数
                if best_instance_mask is not None:
                    intersection = (q_prob_roi * best_instance_mask).sum()
                    q_sum = q_prob_roi.sum()
                    r_sum = best_instance_mask.sum()
                    dice = (2 * intersection + 1e-6) / (q_sum + r_sum + 1e-6)
                    dice_scores_q2r[n] = dice
                else:
                    dice_scores_q2r[n] = 0.0
            
            # 加权累加Query->Render分数
            final_scores += weights[c] * dice_scores_q2r
        
        # ==================== 第二部分：Render -> Query 匹配 ====================
        # 对每个render图提取实例，然后与query匹配
        dice_scores_r2q = torch.zeros(N, device=q_probs.device)
        
        for n in range(N):
            current_render = r_db_descriptors[n]  # [3, H, W]
            
            # RGB转换为颜色ID
            render_2d = (current_render[0] * 65536 + current_render[1] * 256 + current_render[2]).int()
            render_numpy = render_2d.cpu().numpy().astype(np.uint32)
            
            # 识别render图中的所有实例
            unique_colors = np.unique(render_numpy)
            unique_colors = unique_colors[unique_colors > 0]  # 排除背景
            
            if len(unique_colors) == 0:
                dice_scores_r2q[n] = 0.0
                continue
            
            # 先获取所有实例（颜色+连通组件）
            all_instances = []
            for color in unique_colors:
                color_mask = (render_numpy == color).astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(color_mask, connectivity=8)
                
                for label_id in range(1, num_labels):
                    component_mask = (labels == label_id).astype(np.float32)
                    
                    # 计算实例的bbox大小
                    coords = np.where(component_mask > 0)
                    if len(coords[0]) == 0:
                        continue
                    
                    r_y1, r_x1 = coords[0].min(), coords[1].min()
                    r_y2, r_x2 = coords[0].max() + 1, coords[1].max() + 1
                    bbox_area = (r_y2 - r_y1) * (r_x2 - r_x1)
                    
                    all_instances.append({
                        'mask': component_mask,
                        'bbox': (r_y1, r_x1, r_y2, r_x2),
                        'area': bbox_area,
                        'color': color,
                        'label_id': label_id
                    })
            
            # 对所有实例进行智能抽样
            if sampling_ratio < 1.0 and len(all_instances) > 0:
                num_samples = max(1, int(len(all_instances) * sampling_ratio))
                
                # 按bbox面积排序
                all_instances.sort(key=lambda x: x['area'], reverse=True)
                
                # 大部分来自大bbox实例（前70%），少部分随机抽样（后30%）
                large_bbox_ratio = 0.8
                num_large_samples = int(num_samples * large_bbox_ratio)
                num_random_samples = num_samples - num_large_samples
                
                # 选择大bbox实例
                selected_instances = all_instances[:num_large_samples]
                
                # 从剩余实例中随机选择
                if num_random_samples > 0 and len(all_instances) > num_large_samples:
                    remaining_instances = all_instances[num_large_samples:]
                    if len(remaining_instances) >= num_random_samples:
                        random_indices = np.random.choice(len(remaining_instances), num_random_samples, replace=False)
                        selected_instances.extend([remaining_instances[i] for i in random_indices])
                    else:
                        selected_instances.extend(remaining_instances)
                
                all_instances = selected_instances
            
            render_instance_scores = []
            render_instance_weights = []
            # 处理选中的render实例
            for instance in all_instances:
                component_mask = instance['mask']
                r_y1, r_x1, r_y2, r_x2 = instance['bbox']
                
                # 确保bbox在有效范围内
                r_x1 = max(0, min(r_x1, W-1))
                r_y1 = max(0, min(r_y1, H-1))
                r_x2 = max(r_x1+1, min(r_x2, W))
                r_y2 = max(r_y1+1, min(r_y2, H))
                
                # 提取render实例的ROI
                r_instance_roi = torch.from_numpy(component_mask[r_y1:r_y2, r_x1:r_x2]).to(q_probs.device)
                
                # 在对应位置寻找最佳匹配的query实例
                max_iou = 0.0
                best_dice = 0.0
                best_query_idx = -1
                    
                for c in range(C):
                    # 提取query实例在相同位置的ROI
                    q_instance_roi = q_probs[c, r_y1:r_y2, r_x1:r_x2]
                    
                    # 计算IOU
                    intersection = (q_instance_roi * r_instance_roi).sum()
                    union = q_instance_roi.sum() + r_instance_roi.sum() - intersection
                    iou = intersection / (union + 1e-6)
                    
                    if iou > max_iou:
                        max_iou = iou
                        best_query_idx = c
                        # 计算Dice系数
                        q_sum = q_instance_roi.sum()
                        r_sum = r_instance_roi.sum()
                        best_dice = (2 * intersection + 1e-6) / (q_sum + r_sum + 1e-6)
                
                # 记录当前render实例的最佳匹配分数
                if max_iou > 0:
                    render_instance_scores.append(best_dice.item())
                    
                    # 根据weight_type计算权重
                    if weight_type == 'uniform':
                        # uniform权重：所有render实例权重相等
                        instance_weight = 1.0
                    elif weight_type == 'area':
                        # area权重：使用render实例的像素面积
                        instance_weight = r_instance_roi.sum().item()
                    elif weight_type == 'scores' and q_weights is not None and best_query_idx >= 0:
                        # scores权重：使用匹配的query实例的score
                        instance_weight = q_weights[best_query_idx].item()
                    else:
                        # 默认使用area权重
                        instance_weight = r_instance_roi.sum().item()
                        
                    render_instance_weights.append(instance_weight)

            # 计算当前render图的加权平均分数
            if len(render_instance_scores) > 0:
                render_instance_weights = np.array(render_instance_weights)
                render_instance_scores = np.array(render_instance_scores)
                
                # 归一化权重
                if render_instance_weights.sum() > 0:
                    if weight_type == 'uniform':
                        # uniform权重：直接平均
                        dice_scores_r2q[n] = render_instance_scores.mean()
                    else:
                        # area或scores权重：加权平均
                        render_instance_weights = render_instance_weights / render_instance_weights.sum()
                        dice_scores_r2q[n] = (render_instance_scores * render_instance_weights).sum()
                else:
                    dice_scores_r2q[n] = 0.0
            else:
                dice_scores_r2q[n] = 0.0
            # # ============================================================
            # # 处理选中的render实例
            # for instance in all_instances:
            #     component_mask = instance['mask']
            #     r_y1, r_x1, r_y2, r_x2 = instance['bbox']
                
            #     # 确保bbox在有效范围内
            #     r_x1 = max(0, min(r_x1, W-1))
            #     r_y1 = max(0, min(r_y1, H-1))
            #     r_x2 = max(r_x1+1, min(r_x2, W))
            #     r_y2 = max(r_y1+1, min(r_y2, H))
                
            #     # 提取render实例的ROI
            #     r_instance_roi = torch.from_numpy(component_mask[r_y1:r_y2, r_x1:r_x2]).to(q_probs.device)
                
            #     # 在对应位置寻找最佳匹配的query实例
            #     max_iou = 0.0
            #     best_dice = 0.0
                    
            #     for c in range(C):
            #         # 提取query实例在相同位置的ROI
            #         q_instance_roi = q_probs[c, r_y1:r_y2, r_x1:r_x2]
                    
            #         # 计算IOU
            #         intersection = (q_instance_roi * r_instance_roi).sum()
            #         union = q_instance_roi.sum() + r_instance_roi.sum() - intersection
            #         iou = intersection / (union + 1e-6)
                    
            #         if iou > max_iou:
            #             max_iou = iou
            #             # 计算Dice系数
            #             q_sum = q_instance_roi.sum()
            #             r_sum = r_instance_roi.sum()
            #             best_dice = (2 * intersection + 1e-6) / (q_sum + r_sum + 1e-6)
                
            #     # 记录当前render实例的最佳匹配分数
            #     if max_iou > 0:
            #         instance_area = r_instance_roi.sum().item()
            #         render_instance_scores.append(best_dice.item())
            #         render_instance_weights.append(instance_area)
            
            # # 计算当前render图的加权平均分数，权重是区域大小
            # if len(render_instance_scores) > 0:
            #     render_instance_weights = np.array(render_instance_weights)
            #     render_instance_scores = np.array(render_instance_scores)
                
            #     # 归一化权重
            #     if render_instance_weights.sum() > 0:
            #         render_instance_weights = render_instance_weights / render_instance_weights.sum()
            #         dice_scores_r2q[n] = (render_instance_scores * render_instance_weights).sum()
            #     else:
            #         dice_scores_r2q[n] = 0.0
            # else:
            #     dice_scores_r2q[n] = 0.0
            # # ============================================================
        
        # 合并两个方向的分数（可以调整权重比例）
        alpha = 0.5  # Query->Render的权重
        beta = 0.5  # Render->Query的权重
        print("final_scores:",final_scores)
        print("dice_scores_r2q:",dice_scores_r2q)
        final_scores = alpha * final_scores + beta * dice_scores_r2q
        
        return final_scores.cpu()

    def calculate_single_dice_score_double(self, q_probs, q_bboxs, render_image, weight_type='uniform', q_weights=None, sampling_ratio=1.0):
        """
        计算单个渲染图像的双向匹配 Dice Loss 分数
        
        Args:
            q_probs: 查询图的实例概率矩阵 [C, H, W]
            q_bboxs: 每个实例的候选框 [C, 4] 格式为 [x1, y1, x2, y2]
            render_image: 单个渲染图像 [3, H, W] (RGB格式)
            weight_type: 权重计算方式 'uniform'、'area' 或 'scores'
            q_weights: 自定义权重 [C]，当weight_type='scores'时使用
            sampling_ratio: render实例的抽样比例 (0.0-1.0)
        
        Returns:
            score: 单个渲染图的分数 (标量)
        """
        # 确保输入在GPU上
        q_probs = q_probs.cuda()
        q_bboxs = torch.tensor(q_bboxs, dtype=torch.int32).cuda()
        render_image = torch.tensor(render_image, dtype=torch.float32).cuda()
        
        C, H, W = q_probs.shape
        
        # 计算权重
        if weight_type == 'uniform':
            weights = torch.ones(C, device=q_probs.device) / C
        elif weight_type == 'area':
            areas = (q_bboxs[:, 2] - q_bboxs[:, 0]) * (q_bboxs[:, 3] - q_bboxs[:, 1])
            weights = areas.float() / areas.sum().float()
        elif weight_type == 'scores':
            if q_weights is None:
                raise ValueError("当weight_type='scores'时，必须提供q_weights参数")
            weights = q_weights.cuda() if not q_weights.is_cuda else q_weights
            weights = weights / weights.sum()
        else:
            raise ValueError(f"不支持的权重类型: {weight_type}")
        
        final_score = 0.0
        
        # ==================== 第一部分：Query -> Render 匹配 ====================
        q2r_score = 0.0
        for c in range(C):
            x1, y1, x2, y2 = q_bboxs[c]
            
            # 确保坐标在有效范围内
            x1 = max(0, min(x1.item(), W-1))
            y1 = max(0, min(y1.item(), H-1))
            x2 = max(x1+1, min(x2.item(), W))
            y2 = max(y1+1, min(y2.item(), H))
            
            # 提取候选框区域
            q_prob_roi = q_probs[c, y1:y2, x1:x2]  # [roi_h, roi_w]
            r_roi = render_image[:, y1:y2, x1:x2]  # [3, roi_h, roi_w]
            
            # RGB转换为颜色ID
            r_roi_2d = (r_roi[0] * 65536 + r_roi[1] * 256 + r_roi[2]).int()
            roi_numpy = r_roi_2d.cpu().numpy().astype(np.uint32)
            
            # 识别不同颜色
            unique_colors = np.unique(roi_numpy)
            unique_colors = unique_colors[unique_colors > 0]
            
            if len(unique_colors) == 0:
                continue
            
            # 找到最佳匹配的render实例
            max_iou = 0.0
            best_instance_mask = None
            
            for color in unique_colors:
                color_mask = (roi_numpy == color).astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(color_mask, connectivity=8)
                
                for label_id in range(1, num_labels):
                    component_mask = (labels == label_id).astype(np.float32)
                    component_mask_tensor = torch.from_numpy(component_mask).to(q_prob_roi.device)
                    
                    intersection = (q_prob_roi * component_mask_tensor).sum()
                    union = q_prob_roi.sum() + component_mask_tensor.sum() - intersection
                    iou = intersection / (union + 1e-6)
                    
                    if iou > max_iou:
                        max_iou = iou
                        best_instance_mask = component_mask_tensor
                
            # 计算Dice系数
            if best_instance_mask is not None:
                intersection = (q_prob_roi * best_instance_mask).sum()
                q_sum = q_prob_roi.sum()
                r_sum = best_instance_mask.sum()
                dice = (2 * intersection + 1e-6) / (q_sum + r_sum + 1e-6)
                q2r_score += weights[c] * dice
        
        # ==================== 第二部分：Render -> Query 匹配 ====================
        r2q_score = 0.0
        
        # RGB转换为颜色ID
        render_2d = (render_image[0] * 65536 + render_image[1] * 256 + render_image[2]).int()
        render_numpy = render_2d.cpu().numpy().astype(np.uint32)
        
        # 识别render图中的所有实例
        unique_colors = np.unique(render_numpy)
        unique_colors = unique_colors[unique_colors > 0]  # 排除背景
        
        if len(unique_colors) > 0:
            # 获取所有实例（颜色+连通组件）
            all_instances = []
            for color in unique_colors:
                color_mask = (render_numpy == color).astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(color_mask, connectivity=8)
                
                for label_id in range(1, num_labels):
                    component_mask = (labels == label_id).astype(np.float32)
                    
                    # 计算实例的bbox大小
                    coords = np.where(component_mask > 0)
                    if len(coords[0]) == 0:
                        continue
                    
                    r_y1, r_x1 = coords[0].min(), coords[1].min()
                    r_y2, r_x2 = coords[0].max() + 1, coords[1].max() + 1
                    bbox_area = (r_y2 - r_y1) * (r_x2 - r_x1)
                    
                    all_instances.append({
                        'mask': component_mask,
                        'bbox': (r_y1, r_x1, r_y2, r_x2),
                        'area': bbox_area,
                        'color': color,
                        'label_id': label_id
                    })
            
            # 对所有实例进行智能抽样
            if sampling_ratio < 1.0 and len(all_instances) > 0:
                num_samples = max(1, int(len(all_instances) * sampling_ratio))
                
                # 按bbox面积排序
                all_instances.sort(key=lambda x: x['area'], reverse=True)
                
                # 大部分来自大bbox实例，少部分随机抽样
                large_bbox_ratio = 0.8
                num_large_samples = int(num_samples * large_bbox_ratio)
                num_random_samples = num_samples - num_large_samples
                
                # 选择大bbox实例
                selected_instances = all_instances[:num_large_samples]
                
                # 从剩余实例中随机选择
                if num_random_samples > 0 and len(all_instances) > num_large_samples:
                    remaining_instances = all_instances[num_large_samples:]
                    if len(remaining_instances) >= num_random_samples:
                        random_indices = np.random.choice(len(remaining_instances), num_random_samples, replace=False)
                        selected_instances.extend([remaining_instances[i] for i in random_indices])
                    else:
                        selected_instances.extend(remaining_instances)
                
                all_instances = selected_instances
            
            render_instance_scores = []
            render_instance_weights = []
            
            # 处理选中的render实例
            for instance in all_instances:
                component_mask = instance['mask']
                r_y1, r_x1, r_y2, r_x2 = instance['bbox']
                
                # 确保bbox在有效范围内
                r_x1 = max(0, min(r_x1, W-1))
                r_y1 = max(0, min(r_y1, H-1))
                r_x2 = max(r_x1+1, min(r_x2, W))
                r_y2 = max(r_y1+1, min(r_y2, H))
                
                # 提取render实例的ROI
                r_instance_roi = torch.from_numpy(component_mask[r_y1:r_y2, r_x1:r_x2]).to(q_probs.device)
                
                # 在对应位置寻找最佳匹配的query实例
                max_iou = 0.0
                best_dice = 0.0
                best_query_idx = -1
                    
                for c in range(C):
                    # 提取query实例在相同位置的ROI
                    q_instance_roi = q_probs[c, r_y1:r_y2, r_x1:r_x2]
                    
                    # 计算IOU
                    intersection = (q_instance_roi * r_instance_roi).sum()
                    union = q_instance_roi.sum() + r_instance_roi.sum() - intersection
                    iou = intersection / (union + 1e-6)
                    
                    if iou > max_iou:
                        max_iou = iou
                        best_query_idx = c
                        # 计算Dice系数
                        q_sum = q_instance_roi.sum()
                        r_sum = r_instance_roi.sum()
                        best_dice = (2 * intersection + 1e-6) / (q_sum + r_sum + 1e-6)
                
                # 记录当前render实例的最佳匹配分数
                if max_iou > 0:
                    render_instance_scores.append(best_dice.item())
                    
                    # 根据weight_type计算权重
                    if weight_type == 'uniform':
                        instance_weight = 1.0
                    elif weight_type == 'area':
                        instance_weight = r_instance_roi.sum().item()
                    elif weight_type == 'scores' and q_weights is not None and best_query_idx >= 0:
                        instance_weight = q_weights[best_query_idx].item()
                    else:
                        instance_weight = r_instance_roi.sum().item()
                        
                    render_instance_weights.append(instance_weight)

            # 计算当前render图的加权平均分数
            if len(render_instance_scores) > 0:
                render_instance_weights = np.array(render_instance_weights)
                render_instance_scores = np.array(render_instance_scores)
                
                # 归一化权重
                if render_instance_weights.sum() > 0:
                    if weight_type == 'uniform':
                        r2q_score = render_instance_scores.mean()
                    else:
                        render_instance_weights = render_instance_weights / render_instance_weights.sum()
                        r2q_score = (render_instance_scores * render_instance_weights).sum()
        
        # 合并两个方向的分数
        alpha = 0.5  # Query->Render的权重
        beta = 0.5  # Render->Query的权重
        final_score = alpha * q2r_score + beta * r2q_score
        
        return float(final_score)