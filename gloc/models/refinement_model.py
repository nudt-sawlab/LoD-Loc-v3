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
        
        rgb_features = torch.tensor(rgb_features)
        target_features = torch.tensor(target_features)

        batch_size_rgb,_, H, W = rgb_features.shape
        batch_size_target,_, _, _ = target_features.shape


        rgb_features_norm = F.normalize(rgb_features.view(batch_size_rgb, -1), p=2, dim=-1)
        target_features_norm = F.normalize(target_features.view(batch_size_target, -1), p=2, dim=-1)


        similarity_matrix = torch.matmul(rgb_features_norm, target_features_norm.t())  # [8, 72]
        

        most_similar_batch_indices = torch.argmax(similarity_matrix, dim=1)  # [8]

        return similarity_matrix.squeeze()# , most_similar_batch_indices
    
    def calc_euclidean_distance(self, F, M):

        # breakpoint()
        assert F.shape[1:] == M.shape[1:], "特征图 F 的通道、高度和宽度必须与模板图集合 M 的一致"

        if isinstance(F, np.ndarray):
            F = torch.from_numpy(F)
        if isinstance(M, np.ndarray):
            M = torch.from_numpy(M)


        F = F / torch.norm(F, p=2, dim=1, keepdim=True)
        M = M / torch.norm(M, p=2, dim=1, keepdim=True)
        

        distances = torch.norm(F - M, p=2, dim=(1, 2))  # (18)
        

        sorted_distances, indices = torch.sort(distances)
        
        return distances
    
    def find_most_IoU_batch(self, mask_feature, target_features):


        # mask_feature = torch.tensor(mask_feature, dtype=torch.float32).cuda()
        mask_feature = mask_feature.cuda()
        target_features = torch.tensor(target_features, dtype=torch.float32).cuda()
        

        intersection = torch.sum(mask_feature * target_features, dim=(1, 2))
        # (Pdb) intersection
        # tensor([77404.1406, 77404.1406, 76803.7031, 78117.1406, 78131.8906, 78610.9688,
        #         75300.7578, 81332.4531, 77470.8594, 72735.8359, 66357.0000, 69243.9062,
        #         70216.5000, 75826.9062, 74974.0000, 76667.3516, 70993.9375, 75582.0156,
        #         78698.2344, 75618.0547, 75012.8906, 72865.2969, 79060.3438, 69175.1484,
        #         75408.2812, 67718.9531], device='cuda:0')

        union = torch.sum(mask_feature, dim=(1, 2)) + torch.sum(target_features, dim=(1, 2)) - intersection


        iou_scores = intersection / union  # [N]
        # (Pdb) iou_scores
        # tensor([0.6681, 0.6681, 0.6598, 0.6841, 0.6719, 0.6771, 0.6367, 0.7105, 0.6700,
        #         0.6016, 0.5460, 0.5689, 0.5941, 0.6493, 0.6439, 0.6719, 0.5816, 0.6426,
        #         0.6761, 0.6576, 0.6492, 0.6217, 0.6869, 0.5561, 0.6488, 0.5485],
        # device='cuda:0')

        most_similar_batch_indices = torch.argmax(iou_scores)
        return iou_scores.cpu().squeeze()
    
    def find_weighted_IoU_multi_class(self, mask_feature, target_features, class_labels, weights=None):
        mask_feature = mask_feature.cuda()
        target_features = torch.from_numpy(target_features).cuda()
        N, H, W = target_features.shape
        C = len(class_labels)

        if weights is None:
            weights = [1.0 / C] * C
        else:
            weights = torch.tensor(weights, device=mask_feature.device)
            weights = weights / weights.sum()

        iou_scores = torch.zeros(N, device=mask_feature.device)

        for i, cls in enumerate(class_labels):

            mask_bin = (mask_feature == cls).float()  # (1, H, W)
            targets_bin = (target_features == cls).float()  # (N, H, W)


            intersection = torch.sum(mask_bin * targets_bin, dim=(1, 2))
            union = torch.sum(mask_bin, dim=(1, 2)) + torch.sum(targets_bin, dim=(1, 2)) - intersection
            iou_per_class = intersection / (union + 1e-6)

            iou_scores += weights[i] * iou_per_class

        return iou_scores.cpu()
    
    def find_weighted_IoU_from_RGB(self, mask_feature, target_features, weights=None):
        mask_feature = mask_feature#.cuda()
        target_features = target_features#.cuda()  # (N, 3, H, W)
        N, _, H, W = target_features.shape


        unique_colors = torch.unique(target_features.permute(0, 2, 3, 1).reshape(-1, 3), dim=0)
        C = unique_colors.shape[0]

        if weights is None:
            weights = torch.ones(C, device=mask_feature.device) / C
        else:
            weights = torch.tensor(weights, device=mask_feature.device)
            weights = weights / weights.sum()

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

        scores = self.find_weighted_IoU_multi_class(q_feats.unsqueeze(0), r_db_descriptors, class_label)
        #++++++++++++


        #++++++++++++
        preds = torch.argsort(-scores)#, descending=True)
        if get_scores:
            return preds, scores[preds]
        return preds

    def rank_candidates_multimask_twostages(self, q_feats, r_db_descriptors, q_weights=None, topk=10, get_scores=False):

        class_label = [1]
        scores = self.find_weighted_IoU_multi_class(q_feats.unsqueeze(0), r_db_descriptors, class_label)
        topk = min(topk, len(scores))
        topk_indices = torch.argsort(-scores)[:topk]

        if q_weights is not None:
            from tloc.refine_diceloss import batch_dice_loss

            candidate_masks = torch.from_numpy(r_db_descriptors[topk_indices.cpu().numpy()]).float().cuda()  # [topk, H, W]

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
        scores = self.calculate_dice_scores(q_probs, q_weights, r_db_descriptors)
        preds = torch.argsort(-scores)
        if get_scores:
            return preds, scores[preds]
        return preds
    
    def calculate_dice_scores(self, q_probs, q_weights, r_db_descriptors):

        q_probs = q_probs.cuda()
        q_weights = q_weights.cuda()
        r_db_descriptors = torch.tensor(r_db_descriptors, dtype=torch.float32).cuda()
        
        C, H, W = q_probs.shape
        N = r_db_descriptors.shape[0]
        

        q_probs_expanded = q_probs.unsqueeze(0).expand(N, C, H, W)  # [N, C, H, W]
        r_db_expanded = r_db_descriptors.unsqueeze(1)  # [N, 1, H, W]
        

        intersection = (q_probs_expanded * r_db_expanded).sum(dim=(2, 3))  # [N, C]
        

        dice = (2 * intersection + 1e-6) / (
            q_probs_expanded.sum(dim=(2, 3)) + r_db_expanded.sum(dim=(2, 3)) + 1e-6
        )  # [N, C]
        

        weighted_scores = (dice * q_weights.unsqueeze(0)).sum(dim=1)  # [N]
        
        return weighted_scores.cpu()


    def rank_candidates_mask_dice_bboxs(self, q_probs, q_bboxs, r_db_descriptors, weight_type='uniform', q_weights=None, get_scores=False):
        # scores = self.calculate_dice_scores_bboxs(q_probs, q_bboxs, r_db_descriptors, weight_type, q_weights)
        scores = self.calculate_dice_scores_bboxs_double(q_probs, q_bboxs, r_db_descriptors, weight_type, q_weights, sampling_ratio=1)
        preds = torch.argsort(-scores)
        if get_scores:
            return preds, scores[preds]
        return preds

    def calculate_dice_scores_bboxs(self, q_probs, q_bboxs, r_db_descriptors, weight_type='uniform', q_weights=None):

        q_probs = q_probs.cuda()
        q_bboxs = torch.tensor(q_bboxs, dtype=torch.int32).cuda()
        r_db_descriptors = torch.tensor(r_db_descriptors, dtype=torch.float32).cuda()
        # breakpoint()
        C, H, W = q_probs.shape
        N = r_db_descriptors.shape[0]
        # breakpoint()

        if weight_type == 'uniform':
            weights = torch.ones(C, device=q_probs.device) / C
        elif weight_type == 'area':

            areas = (q_bboxs[:, 2] - q_bboxs[:, 0]) * (q_bboxs[:, 3] - q_bboxs[:, 1])
            weights = areas.float() / areas.sum().float()
        elif weight_type == 'scores':

            if q_weights is None:
                raise ValueError("当weight_type='scores'时，必须提供q_weights参数")
            weights = q_weights.cuda() if not q_weights.is_cuda else q_weights
        else:
            raise ValueError(f"不支持的权重类型: {weight_type}")
        

        final_scores = torch.zeros(N, device=q_probs.device)
        

        for c in range(C):

            x1, y1, x2, y2 = q_bboxs[c]
            

            x1 = max(0, min(x1.item(), W-1))
            y1 = max(0, min(y1.item(), H-1))
            x2 = max(x1+1, min(x2.item(), W))
            y2 = max(y1+1, min(y2.item(), H))
            

            q_prob_roi = q_probs[c, y1:y2, x1:x2]  # [roi_h, roi_w]
            # r_db_roi = r_db_descriptors[:, y1:y2, x1:x2]  # [N, roi_h, roi_w]

            r_db_roi = r_db_descriptors[:, :, y1:y2, x1:x2]
            # breakpoint()
            # ========================================================================================

            dice_scores = torch.zeros(N, device=q_probs.device)
            
            for n in range(N):

                current_roi = r_db_roi[n]  # [3, roi_h, roi_w]
                

                current_roi_2d = (current_roi[0] * 65536 + current_roi[1] * 256 + current_roi[2]).int()
                


                roi_numpy = current_roi_2d.cpu().numpy().astype(np.uint32)
                

                unique_colors = np.unique(roi_numpy)
                unique_colors = unique_colors[unique_colors > 0]
                
                if len(unique_colors) == 0:
                    dice_scores[n] = 0.0
                    continue
                

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
                    

                if best_instance_mask is not None:
                    intersection = (q_prob_roi * best_instance_mask).sum()
                    q_sum = q_prob_roi.sum()
                    r_sum = best_instance_mask.sum()
                    dice = (2 * intersection + 1e-6) / (q_sum + r_sum + 1e-6)
                    dice_scores[n] = dice
                else:
                    dice_scores[n] = 0.0
            

            final_scores += weights[c] * dice_scores
            # ========================================================================================

            # intersection = (q_prob_roi.unsqueeze(0) * r_db_roi).sum(dim=(1, 2))  # [N]
            

            # q_sum = q_prob_roi.sum()
            # r_sum = r_db_roi.sum(dim=(1, 2))  # [N]
            
            # dice = (2 * intersection + 1e-6) / (q_sum + r_sum + 1e-6)  # [N]
            

            # final_scores += weights[c] * dice
        
        return final_scores.cpu()
# =======================================================================
    # def compute_instance_diou(self, mask1, mask2):
    #     """

    #     Args:


    #     Returns:

    #     """

    #     intersection = torch.logical_and(mask1, mask2).sum()
    #     union = torch.logical_or(mask1, mask2).sum()
    #     iou = intersection / (union + 1e-6)


    #     coords1 = mask1.nonzero()
    #     coords2 = mask2.nonzero()
    #     if coords1.shape[0] == 0 or coords2.shape[0] == 0:

        

    #     min_xy = torch.min(
    #         torch.cat([coords1.min(dim=0)[0], coords2.min(dim=0)[0]]), 
    #         dim=0
    #     )[0]
    #     max_xy = torch.max(
    #         torch.cat([coords1.max(dim=0)[0], coords2.max(dim=0)[0]]),
    #         dim=0
    #     )[0]
        


        

    #     center1 = coords1.float().mean(dim=0)
    #     center2 = coords2.float().mean(dim=0)

        

    #     diou = iou - (d / (c + 1e-6))
    #     return diou.item()

    # def compute_instance_giou(self, mask1, mask2):
    #     """

    #     Args:


    #     Returns:

    #     """
    #     intersection = torch.logical_and(mask1, mask2).sum()
    #     union = torch.logical_or(mask1, mask2).sum()
        

    #     coords = torch.cat([mask1.nonzero(), mask2.nonzero()])

    #         return 0.0
        
    #     min_coord = coords.min(dim=0)[0]
    #     max_coord = coords.max(dim=0)[0]
    #     convex_area = (max_coord[0] - min_coord[0] + 1) * (max_coord[1] - min_coord[1] + 1)
        

    #     iou = intersection / (union + 1e-6)
    #     giou = iou - (convex_area - union) / (convex_area + 1e-6)
    #     return giou.item()

    # # def instance_matching(self, query_instances, render_instances):
    # #     """

    # #     Args:


    # #     Returns:



    # #     """

    # #     cost_matrix = np.zeros((len(query_instances), len(render_instances)))
    # #     for i, (q_mask, _) in enumerate(query_instances):
    # #         for j, r_mask in enumerate(render_instances):

        

    # #     row_ind, col_ind = linear_sum_assignment(cost_matrix)
        

    # #     matched_pairs = []
    # #     for i, j in zip(row_ind, col_ind):
    # #         giou = 1 - cost_matrix[i,j]

    # #             matched_pairs.append((i, j, giou))
    # #         # matched_pairs.append((i, j, giou))
        

    # #     matched_q = set(i for i,_,_ in matched_pairs)
    # #     matched_r = set(j for _,j,_ in matched_pairs)
    # #     un_matched_q = len(query_instances) - len(matched_q)
    # #     un_matched_r = len(render_instances) - len(matched_r)
        

    # #     total_score = sum(giou * query_instances[i][1] for i,j,giou in matched_pairs)
    # #     return total_score, un_matched_q, un_matched_r
    
    # def instance_matching(self, query_instances, render_instances):
    #     """

    #     Args:
    #         query_instances: [(q_mask, q_weight), ...] 
    #         render_instances: [r_mask, ...]
    #     """
    #     n_query = len(query_instances)
    #     n_render = len(render_instances)
        

    #     max_dim = max(n_query, n_render)



    #     for i, (q_mask, _) in enumerate(query_instances):
    #         for j, r_mask in enumerate(render_instances):
    #             giou = self.compute_instance_giou(q_mask, r_mask)




    #     if n_query < max_dim:

    #     if n_render < max_dim:



    #     row_ind, col_ind = linear_sum_assignment(cost_matrix)
        

    #     matched_pairs = []
    #     valid_indices = (row_ind < n_query) & (col_ind < n_render)
    #     for i, j in zip(row_ind[valid_indices], col_ind[valid_indices]):
    #         giou = 1 - cost_matrix[i, j]

    #             matched_pairs.append((i, j, giou))


    #     matched_q = set(i for i,_,_ in matched_pairs)
    #     matched_r = set(j for _,j,_ in matched_pairs)
        
    #     return (
    #         sum(giou * query_instances[i][1] for i, _, giou in matched_pairs),
    #         len(query_instances) - len(matched_q),
    #         len(render_instances) - len(matched_r)
    #     )


    # def rank_candidates_instance_giou(self, q_feats, r_db_descriptors, get_scores=False):
    #     """

    #     Args:


    #     """

    #     query_instances = self.extract_instances(q_feats)  # [(mask, weight), ...]
        
    #     scores = []
    #     for r_mask in r_db_descriptors:

    #         render_instances = self.extract_instances(r_mask)
            

    #         # matched_score, un_q, un_r = self.instance_matching(query_instances, render_instances)

    #         matched_score, un_q, un_r = self.instance_matching(query_instances,render_instances)

    #         penalty = (un_q * 0.2) + (un_r * 0.1 * (len(render_instances)/len(query_instances)))
    #         final_score = matched_score - penalty
            
    #         # scores.append(final_score)


    #         # final_score = matched_score - penalty
            
    #         scores.append(final_score)
        
    #     scores = torch.tensor(scores)
    #     preds = torch.argsort(-scores)
    #     return (preds, scores[preds]) if get_scores else preds

    # def extract_instances(self, mask):
    #     """

    #     Args:

    #     Returns:


    #     """
    #     instances = []
    #     instance_ids = torch.unique(mask)
    #     instance_ids = instance_ids[instance_ids != 0]
        
    #     total_area = sum((mask == iid).sum().item() for iid in instance_ids)
    #     for iid in instance_ids:
    #         ins_mask = (mask == iid)
    #         area = ins_mask.sum().item()

    #         instances.append((ins_mask, weight))
    #     return instances

    def calculate_dice_scores_bboxs_double(self, q_probs, q_bboxs, r_db_descriptors, weight_type='uniform', q_weights=None, sampling_ratio=1.0):

        q_probs = q_probs.cuda()
        q_bboxs = torch.tensor(q_bboxs, dtype=torch.int32).cuda()
        r_db_descriptors = torch.tensor(r_db_descriptors, dtype=torch.float32).cuda()
        
        C, H, W = q_probs.shape
        N = r_db_descriptors.shape[0]
        

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
        

        final_scores = torch.zeros(N, device=q_probs.device)
        


        for c in range(C):
            x1, y1, x2, y2 = q_bboxs[c]
            

            x1 = max(0, min(x1.item(), W-1))
            y1 = max(0, min(y1.item(), H-1))
            x2 = max(x1+1, min(x2.item(), W))
            y2 = max(y1+1, min(y2.item(), H))
            

            q_prob_roi = q_probs[c, y1:y2, x1:x2]  # [roi_h, roi_w]
            r_db_roi = r_db_descriptors[:, :, y1:y2, x1:x2]  # [N, 3, roi_h, roi_w]
            
            dice_scores_q2r = torch.zeros(N, device=q_probs.device)
            
            for n in range(N):
                current_roi = r_db_roi[n]  # [3, roi_h, roi_w]
                

                current_roi_2d = (current_roi[0] * 65536 + current_roi[1] * 256 + current_roi[2]).int()
                roi_numpy = current_roi_2d.cpu().numpy().astype(np.uint32)
                

                unique_colors = np.unique(roi_numpy)
                unique_colors = unique_colors[unique_colors > 0]
                
                if len(unique_colors) == 0:
                    dice_scores_q2r[n] = 0.0
                    continue
                

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
                    

                if best_instance_mask is not None:
                    intersection = (q_prob_roi * best_instance_mask).sum()
                    q_sum = q_prob_roi.sum()
                    r_sum = best_instance_mask.sum()
                    dice = (2 * intersection + 1e-6) / (q_sum + r_sum + 1e-6)
                    dice_scores_q2r[n] = dice
                else:
                    dice_scores_q2r[n] = 0.0
            

            final_scores += weights[c] * dice_scores_q2r
        


        dice_scores_r2q = torch.zeros(N, device=q_probs.device)
        
        for n in range(N):
            current_render = r_db_descriptors[n]  # [3, H, W]
            

            render_2d = (current_render[0] * 65536 + current_render[1] * 256 + current_render[2]).int()
            render_numpy = render_2d.cpu().numpy().astype(np.uint32)
            

            unique_colors = np.unique(render_numpy)
            unique_colors = unique_colors[unique_colors > 0]
            
            if len(unique_colors) == 0:
                dice_scores_r2q[n] = 0.0
                continue
            

            all_instances = []
            for color in unique_colors:
                color_mask = (render_numpy == color).astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(color_mask, connectivity=8)
                
                for label_id in range(1, num_labels):
                    component_mask = (labels == label_id).astype(np.float32)
                    

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
            

            if sampling_ratio < 1.0 and len(all_instances) > 0:
                num_samples = max(1, int(len(all_instances) * sampling_ratio))
                

                all_instances.sort(key=lambda x: x['area'], reverse=True)
                

                large_bbox_ratio = 0.8
                num_large_samples = int(num_samples * large_bbox_ratio)
                num_random_samples = num_samples - num_large_samples
                

                selected_instances = all_instances[:num_large_samples]
                

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

            for instance in all_instances:
                component_mask = instance['mask']
                r_y1, r_x1, r_y2, r_x2 = instance['bbox']
                

                r_x1 = max(0, min(r_x1, W-1))
                r_y1 = max(0, min(r_y1, H-1))
                r_x2 = max(r_x1+1, min(r_x2, W))
                r_y2 = max(r_y1+1, min(r_y2, H))
                

                r_instance_roi = torch.from_numpy(component_mask[r_y1:r_y2, r_x1:r_x2]).to(q_probs.device)
                

                max_iou = 0.0
                best_dice = 0.0
                best_query_idx = -1
                    
                for c in range(C):

                    q_instance_roi = q_probs[c, r_y1:r_y2, r_x1:r_x2]
                    

                    intersection = (q_instance_roi * r_instance_roi).sum()
                    union = q_instance_roi.sum() + r_instance_roi.sum() - intersection
                    iou = intersection / (union + 1e-6)
                    
                    if iou > max_iou:
                        max_iou = iou
                        best_query_idx = c

                        q_sum = q_instance_roi.sum()
                        r_sum = r_instance_roi.sum()
                        best_dice = (2 * intersection + 1e-6) / (q_sum + r_sum + 1e-6)
                

                if max_iou > 0:
                    render_instance_scores.append(best_dice.item())
                    

                    if weight_type == 'uniform':

                        instance_weight = 1.0
                    elif weight_type == 'area':

                        instance_weight = r_instance_roi.sum().item()
                    elif weight_type == 'scores' and q_weights is not None and best_query_idx >= 0:

                        instance_weight = q_weights[best_query_idx].item()
                    else:

                        instance_weight = r_instance_roi.sum().item()
                        
                    render_instance_weights.append(instance_weight)


            if len(render_instance_scores) > 0:
                render_instance_weights = np.array(render_instance_weights)
                render_instance_scores = np.array(render_instance_scores)
                

                if render_instance_weights.sum() > 0:
                    if weight_type == 'uniform':

                        dice_scores_r2q[n] = render_instance_scores.mean()
                    else:

                        render_instance_weights = render_instance_weights / render_instance_weights.sum()
                        dice_scores_r2q[n] = (render_instance_scores * render_instance_weights).sum()
                else:
                    dice_scores_r2q[n] = 0.0
            else:
                dice_scores_r2q[n] = 0.0
            # # ============================================================

            # for instance in all_instances:
            #     component_mask = instance['mask']
            #     r_y1, r_x1, r_y2, r_x2 = instance['bbox']
                

            #     r_x1 = max(0, min(r_x1, W-1))
            #     r_y1 = max(0, min(r_y1, H-1))
            #     r_x2 = max(r_x1+1, min(r_x2, W))
            #     r_y2 = max(r_y1+1, min(r_y2, H))
                

            #     r_instance_roi = torch.from_numpy(component_mask[r_y1:r_y2, r_x1:r_x2]).to(q_probs.device)
                

            #     max_iou = 0.0
            #     best_dice = 0.0
                    
            #     for c in range(C):

            #         q_instance_roi = q_probs[c, r_y1:r_y2, r_x1:r_x2]
                    

            #         intersection = (q_instance_roi * r_instance_roi).sum()
            #         union = q_instance_roi.sum() + r_instance_roi.sum() - intersection
            #         iou = intersection / (union + 1e-6)
                    
            #         if iou > max_iou:
            #             max_iou = iou

            #             q_sum = q_instance_roi.sum()
            #             r_sum = r_instance_roi.sum()
            #             best_dice = (2 * intersection + 1e-6) / (q_sum + r_sum + 1e-6)
                

            #     if max_iou > 0:
            #         instance_area = r_instance_roi.sum().item()
            #         render_instance_scores.append(best_dice.item())
            #         render_instance_weights.append(instance_area)
            

            # if len(render_instance_scores) > 0:
            #     render_instance_weights = np.array(render_instance_weights)
            #     render_instance_scores = np.array(render_instance_scores)
                

            #     if render_instance_weights.sum() > 0:
            #         render_instance_weights = render_instance_weights / render_instance_weights.sum()
            #         dice_scores_r2q[n] = (render_instance_scores * render_instance_weights).sum()
            #     else:
            #         dice_scores_r2q[n] = 0.0
            # else:
            #     dice_scores_r2q[n] = 0.0
            # # ============================================================
        

        alpha = 0.5
        beta = 0.5
        print("final_scores:",final_scores)
        print("dice_scores_r2q:",dice_scores_r2q)
        final_scores = alpha * final_scores + beta * dice_scores_r2q
        
        return final_scores.cpu()

    def calculate_single_dice_score_double(self, q_probs, q_bboxs, render_image, weight_type='uniform', q_weights=None, sampling_ratio=1.0):

        q_probs = q_probs.cuda()
        q_bboxs = torch.tensor(q_bboxs, dtype=torch.int32).cuda()
        render_image = torch.tensor(render_image, dtype=torch.float32).cuda()
        
        C, H, W = q_probs.shape
        

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
        

        q2r_score = 0.0
        for c in range(C):
            x1, y1, x2, y2 = q_bboxs[c]
            

            x1 = max(0, min(x1.item(), W-1))
            y1 = max(0, min(y1.item(), H-1))
            x2 = max(x1+1, min(x2.item(), W))
            y2 = max(y1+1, min(y2.item(), H))
            

            q_prob_roi = q_probs[c, y1:y2, x1:x2]  # [roi_h, roi_w]
            r_roi = render_image[:, y1:y2, x1:x2]  # [3, roi_h, roi_w]
            

            r_roi_2d = (r_roi[0] * 65536 + r_roi[1] * 256 + r_roi[2]).int()
            roi_numpy = r_roi_2d.cpu().numpy().astype(np.uint32)
            

            unique_colors = np.unique(roi_numpy)
            unique_colors = unique_colors[unique_colors > 0]
            
            if len(unique_colors) == 0:
                continue
            

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
                

            if best_instance_mask is not None:
                intersection = (q_prob_roi * best_instance_mask).sum()
                q_sum = q_prob_roi.sum()
                r_sum = best_instance_mask.sum()
                dice = (2 * intersection + 1e-6) / (q_sum + r_sum + 1e-6)
                q2r_score += weights[c] * dice
        

        r2q_score = 0.0
        

        render_2d = (render_image[0] * 65536 + render_image[1] * 256 + render_image[2]).int()
        render_numpy = render_2d.cpu().numpy().astype(np.uint32)
        

        unique_colors = np.unique(render_numpy)
        unique_colors = unique_colors[unique_colors > 0]
        
        if len(unique_colors) > 0:

            all_instances = []
            for color in unique_colors:
                color_mask = (render_numpy == color).astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(color_mask, connectivity=8)
                
                for label_id in range(1, num_labels):
                    component_mask = (labels == label_id).astype(np.float32)
                    

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
            

            if sampling_ratio < 1.0 and len(all_instances) > 0:
                num_samples = max(1, int(len(all_instances) * sampling_ratio))
                

                all_instances.sort(key=lambda x: x['area'], reverse=True)
                

                large_bbox_ratio = 0.8
                num_large_samples = int(num_samples * large_bbox_ratio)
                num_random_samples = num_samples - num_large_samples
                

                selected_instances = all_instances[:num_large_samples]
                

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
            

            for instance in all_instances:
                component_mask = instance['mask']
                r_y1, r_x1, r_y2, r_x2 = instance['bbox']
                

                r_x1 = max(0, min(r_x1, W-1))
                r_y1 = max(0, min(r_y1, H-1))
                r_x2 = max(r_x1+1, min(r_x2, W))
                r_y2 = max(r_y1+1, min(r_y2, H))
                

                r_instance_roi = torch.from_numpy(component_mask[r_y1:r_y2, r_x1:r_x2]).to(q_probs.device)
                

                max_iou = 0.0
                best_dice = 0.0
                best_query_idx = -1
                    
                for c in range(C):

                    q_instance_roi = q_probs[c, r_y1:r_y2, r_x1:r_x2]
                    

                    intersection = (q_instance_roi * r_instance_roi).sum()
                    union = q_instance_roi.sum() + r_instance_roi.sum() - intersection
                    iou = intersection / (union + 1e-6)
                    
                    if iou > max_iou:
                        max_iou = iou
                        best_query_idx = c

                        q_sum = q_instance_roi.sum()
                        r_sum = r_instance_roi.sum()
                        best_dice = (2 * intersection + 1e-6) / (q_sum + r_sum + 1e-6)
                

                if max_iou > 0:
                    render_instance_scores.append(best_dice.item())
                    

                    if weight_type == 'uniform':
                        instance_weight = 1.0
                    elif weight_type == 'area':
                        instance_weight = r_instance_roi.sum().item()
                    elif weight_type == 'scores' and q_weights is not None and best_query_idx >= 0:
                        instance_weight = q_weights[best_query_idx].item()
                    else:
                        instance_weight = r_instance_roi.sum().item()
                        
                    render_instance_weights.append(instance_weight)


            if len(render_instance_scores) > 0:
                render_instance_weights = np.array(render_instance_weights)
                render_instance_scores = np.array(render_instance_scores)
                

                if render_instance_weights.sum() > 0:
                    if weight_type == 'uniform':
                        r2q_score = render_instance_scores.mean()
                    else:
                        render_instance_weights = render_instance_weights / render_instance_weights.sum()
                        r2q_score = (render_instance_scores * render_instance_weights).sum()
        

        alpha = 0.5
        beta = 0.5
        final_score = alpha * q2r_score + beta * r2q_score
        
        return float(final_score)