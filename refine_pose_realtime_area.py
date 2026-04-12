import logging
import os
import sys
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data.dataset import Subset
import torchvision.transforms as T
import commons
from parse_args import parse_args
from path_configs import get_path_conf
from gloc import initialization
from gloc.datasets import get_dataset, get_transform
from gloc.utils import utils, visualization, qvec2rotmat, rotmat2qvec
from gloc.resamplers import get_protocol
from configs import get_config
from utils_tool import get_t_euler
from RealTime_render import RealTime_render
import cupy as cp
import cv2
from PIL import Image
# 设置matplotlib为非交互式后端，避免Qt插件问题
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端


def calculate_batch_dice_score_double_improved(q_probs, q_bboxs, render_images_batch, weight_type='area', q_weights=None, threshold=0.1):

    assert isinstance(q_probs, torch.Tensor) and q_probs.is_cuda
    assert isinstance(q_bboxs, torch.Tensor) and q_bboxs.is_cuda
    assert isinstance(render_images_batch, torch.Tensor) and render_images_batch.is_cuda

    # 确保所有张量都是float32精度，避免half/float混合运算错误
    q_probs = q_probs.float()
    q_bboxs = q_bboxs.float()
    render_images_batch = render_images_batch.float()

    C, H, W = q_probs.shape

    if weight_type == 'uniform':
        weights = torch.ones(C, device=q_probs.device, dtype=torch.float32) / float(C)
    elif weight_type == 'area':
        areas = (q_bboxs[:, 2] - q_bboxs[:, 0]).float() * (q_bboxs[:, 3] - q_bboxs[:, 1]).float()
        weights = areas / (areas.sum() + 1e-6)
    elif weight_type == 'scores':
        if q_weights is None:
            raise ValueError("当weight_type='scores'时，必须提供q_weights参数")
        weights = q_weights.float() if q_weights.is_cuda else q_weights.cuda().float()
        weights = weights / (weights.sum() + 1e-6)
    else:
        raise ValueError(f"不支持的权重类型: {weight_type}")
    
    # 预计算query各实例像素和（全图，不做ROI裁剪）
    q_sums = q_probs.sum(dim=(1, 2))  # [C]

    B = render_images_batch.shape[0]
    out_scores = torch.zeros(B, device=q_probs.device, dtype=torch.float32)

    eps = 1e-6

    # 优化9：批量化颜色ID映射，减少循环开销
    # 将整个batch一次性转换为颜色ID
    render_2d_batch = (render_images_batch[:, 0] * 65536 + render_images_batch[:, 1] * 256 + render_images_batch[:, 2]).to(torch.int32)
    
    for b in range(B):
        render_2d = render_2d_batch[b]

        unique_colors = torch.unique(render_2d)
        unique_colors = unique_colors[unique_colors > 0]
        if unique_colors.numel() == 0:
            out_scores[b] = 0.0
            continue

        # 颜色掩码 [K, H, W]
        color_vals = unique_colors.view(-1, 1, 1)
        color_masks = (render_2d.unsqueeze(0) == color_vals).to(torch.float32)

        # 交并比成分
        # intersections: [K, C] = sum_hw(color_masks[K,H,W] * q_probs[C,H,W])
        intersections = torch.einsum('khw,chw->kc', color_masks, q_probs)
        r_sums = color_masks.sum(dim=(1, 2))  # [K]

        denom = q_sums.unsqueeze(0) + r_sums.unsqueeze(1) + eps  # [K, C]
        dice_kc = (2.0 * intersections + eps) / denom

        # Query->Render: 每个c取颜色维度最大
        q2r_per_c = dice_kc.max(dim=0).values  # [C]
        q2r_score = (q2r_per_c * weights).sum()

        out_scores[b] = q2r_score
        

    return out_scores

# 以下是原始main函数的完整实现，保持不变
def main(args):
    # 设置是否保存渲染结果的选项
    if not hasattr(args, 'save_renders'):
        args.save_renders = False
    
    commons.make_deterministic(args.seed)
    commons.setup_logging(args.save_dir, console="info")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")
    
    # 创建渲染结果保存目录
    if args.save_renders:
        render_save_dir = os.path.join(args.save_dir, 'renders')
        os.makedirs(render_save_dir, exist_ok=True)
        logging.info(f"Rendered images will be saved to {render_save_dir}")
    
    paths_conf = get_path_conf(args.colmap_res, args.mesh)
    exp_config = get_config(args.name)
    
    # 使用配置文件中的参数来更新args
    if exp_config and not isinstance(exp_config, type(NotImplementedError)):
        # 取第一个配置（通常是最主要的配置）
        main_config = exp_config[0]
        
        # 更新关键参数
        if isinstance(main_config, dict):
            if 'beams' in main_config:
                args.beams = main_config['beams']
            if 'steps' in main_config:
                args.steps = main_config['steps']
            if 'N' in main_config:
                args.N = main_config['N']
            if 'M' in main_config:
                args.M = main_config['M']
            if 'protocol' in main_config:
                args.protocol = main_config['protocol']
            if 'center_std' in main_config:
                args.center_std = main_config['center_std']
            if 'teta' in main_config:
                args.teta = main_config['teta']
            if 'gamma' in main_config:
                args.gamma = main_config['gamma']
            if 'feat_model' in main_config:
                args.feat_model = main_config['feat_model']
            if 'res' in main_config:
                args.res = main_config['res']
            if 'colmap_res' in main_config:
                args.colmap_res = main_config['colmap_res']
            if 'feat_level' in main_config:
                args.feat_level = main_config['feat_level']
            
            logging.info(f"Updated args with config: beams={args.beams}, steps={args.steps}, N={args.N}, M={args.M}, protocol={args.protocol}")
            logging.info(f"Config parameters: center_std={args.center_std}, teta={args.teta}, gamma={args.gamma}")
    
    # 确保配置参数为整数类型
    args.N = int(args.N) if hasattr(args, 'N') else 52
    args.beams = int(args.beams) if hasattr(args, 'beams') else 2
    args.M = int(args.M) if hasattr(args, 'M') else 2
    args.steps = int(args.steps) if hasattr(args, 'steps') else 1
    
    # 初始化渲染器
    render2loc = RealTime_render(args.render_config)
    
    # 获取数据集
    DS = args.name
    res = args.res
    colmap_dir = paths_conf[DS]['colmap']
    if isinstance(colmap_dir, list):
        colmap_dir = colmap_dir[0]  # 如果是列表，取第一个元素
    transform = get_transform(args, colmap_dir)
    pose_dataset = get_dataset(DS, paths_conf[DS], transform)
    
    # 获取query掩码
    queries_subset = Subset(pose_dataset, pose_dataset.q_frames_idxs)
    q_descriptors = get_query_masks(queries_subset, transform)

    # 在 main 函数中，移除或修改 DenseFeaturesRefiner 的初始化部分
    fine_model = None
    if args.use_dice_evaluation:
        # 由于我们已经将函数集成到脚本中，不再需要 DenseFeaturesRefiner 类
        fine_model = True  # 简单的标志，表示启用 Dice 评估
        logging.info("Dice evaluation enabled with improved integrated function")
    
    # 获取初始预测和recall值
    first_step, all_pred_t, all_pred_R, scores = initialization.init_refinement(args, pose_dataset)
    
    # 检查初始预测是否为None
    if all_pred_t is None or all_pred_R is None:
        logging.error("Failed to get initial predictions")
        return
    
    # 对每个query单独处理
    final_scores = scores  # 从init_refinement获得的字典格式scores
    total_queries = len(pose_dataset.q_frames_idxs)
    
    # 优化3：分批加载query数据到GPU内存，避免内存溢出
    batch_size = 100  # 每批处理100个query
    total_batches = (total_queries + batch_size - 1) // batch_size  # 向上取整
    logging.info(f"将分{total_batches}个批次处理{total_queries}个query，每批最多{batch_size}个")
    
    # 收集所有query的最终位姿 - 使用正确的numpy数组格式
    max_candidates = args.N  # 最大候选数量
    all_final_pred_t = np.zeros((total_queries, max_candidates, 3))
    all_final_pred_R = np.zeros((total_queries, max_candidates, 3, 3))
    all_final_scores = np.zeros((total_queries, max_candidates))
    
    # 分批处理所有query
    for batch_idx in range(total_batches):
        # 计算当前批次的query索引范围
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_queries)
        batch_indices = list(range(batch_start, batch_end))
        
        processed_queries = batch_start
        remaining_queries = total_queries - processed_queries
        progress_percent = (processed_queries / total_queries) * 100
        logging.info(f"开始处理批次 {batch_idx + 1}/{total_batches}: query {batch_start}-{batch_end-1} ({len(batch_indices)}个)")
        logging.info(f"总体进度: {processed_queries}/{total_queries} ({progress_percent:.1f}%), 剩余 {remaining_queries} 个query")
        
        # 加载当前批次的数据到GPU
        batch_data = {}
        if args.use_dice_evaluation and fine_model:
            batch_data = load_batch_query_data(batch_indices, pose_dataset, args)
        
        # 处理当前批次的每个query
        for q_idx in tqdm(batch_indices, desc=f"Batch {batch_idx+1}/{total_batches}"):
            # 获取该query的信息
            idx = pose_dataset.q_frames_idxs[q_idx]
            q_name = pose_dataset.get_basename(idx)
            q_name_clean = q_name[6:]  # 去掉 'query_' 前缀
            q_key_name = os.path.splitext(pose_dataset.images[idx].name)[0]
            
            # 获取相机内参
            if hasattr(pose_dataset, 'intrinsics') and q_key_name in pose_dataset.intrinsics:
                intrinsic_data = pose_dataset.intrinsics[q_key_name]
                camera_K = intrinsic_data['K']
                w = intrinsic_data['w']
                h = intrinsic_data['h']
            else:
                logging.warning(f"Cannot get intrinsics for {q_key_name}, using default values")
                camera_K = np.eye(3)
                w, h = 720, 480
                # w, h = 602, 448
                # w, h = 640, 360

            q_mask = q_descriptors[q_idx]
            # 该query的当前最佳位姿
            query_pred_t = all_pred_t[q_idx]
            query_pred_R = all_pred_R[q_idx]
            
            # 获取当前批次中预加载的数据
            preloaded_query_data = batch_data.get(q_idx, None)
            
            # 对该query进行迭代优化
            query_scores, final_pred_t, final_pred_R, query_scores_array = process_single_query(
                q_idx, q_name_clean, q_mask, camera_K, w, h,
                query_pred_t, query_pred_R,
                args, render2loc, fine_model,
                first_step, preloaded_query_data  # 传递批次中预加载的数据
            )
            
            # 收集最终位姿
            if final_pred_t is not None and final_pred_R is not None and query_scores_array is not None:
                # 对最终步骤的候选进行排序
                sorted_pred_R, sorted_pred_t, flat_preds, sorted_scores = sort_candidates_by_score(
                    query_scores_array, final_pred_t, final_pred_R
                )
                
                # 存储排序后的候选（最多max_candidates个）
                n_candidates = min(len(sorted_pred_t), max_candidates)
                all_final_pred_t[q_idx, :n_candidates] = sorted_pred_t[:n_candidates]
                all_final_pred_R[q_idx, :n_candidates] = sorted_pred_R[:n_candidates]
                all_final_scores[q_idx, :n_candidates] = sorted_scores[:n_candidates]
                
                # 如果候选不够，用最佳候选填充
                if n_candidates < max_candidates:
                    for i in range(n_candidates, max_candidates):
                        all_final_pred_t[q_idx, i] = sorted_pred_t[0]
                        all_final_pred_R[q_idx, i] = sorted_pred_R[0]
                        all_final_scores[q_idx, i] = sorted_scores[0]
            else:
                # 如果没有最终位姿，使用初始位姿
                logging.warning(f"No final poses for query {q_idx}, using initial poses")
                for i in range(max_candidates):
                    all_final_pred_t[q_idx, i] = all_pred_t[q_idx, 0, 0]
                    all_final_pred_R[q_idx, i] = all_pred_R[q_idx, 0, 0]
                    all_final_scores[q_idx, i] = 0.0
            
            # 更新总体分数
            final_scores = utils.update_scores(final_scores, query_scores)
        
        # 当前批次处理完成，清理GPU内存
        if batch_data:
            clear_batch_data(batch_data)
            logging.info(f"批次 {batch_idx + 1}/{total_batches} 处理完成并清理内存")
    
    # 所有query处理完成后，计算整体评估结果
    logging.info("="*50)
    logging.info("COMPUTING FINAL EVALUATION FOR ALL QUERIES")
    logging.info("="*50)
    
    try:
        # 获取ground truth
        all_true_t, all_true_R = pose_dataset.get_q_poses()
        
        # 使用正确的数据格式进行评估
        final_pred_t = all_final_pred_t[:, 0, :]  # (n_queries, 3)
        final_pred_R = all_final_pred_R[:, 0, :, :]  # (n_queries, 3, 3)
        
        # 确保数据格式正确
        final_pred_t = final_pred_t.reshape(total_queries, 1, 3)  # (n_queries, 1, 3)
        final_pred_R = final_pred_R.reshape(total_queries, 1, 3, 3)  # (n_queries, 1, 3, 3)
        
        # 计算所有query的误差
        errors_t, errors_R = utils.get_all_errors_first_estimate(
            all_true_t, all_true_R, final_pred_t, final_pred_R
        )
        
        # 为了使用eval_poses_top_n，需要扩展维度
        all_errors_t = errors_t.reshape(total_queries, 1)  # (n_queries, 1)
        all_errors_R = errors_R.reshape(total_queries, 1)  # (n_queries, 1)
        
        # 调用eval_poses_top_n显示最终评估结果
        result_str, results = utils.eval_poses_top_n(
            all_errors_t, all_errors_R, descr=f'step {args.steps}'
        )
        
        logging.info("FINAL EVALUATION RESULTS:")
        logging.info(result_str)
        
        # 保存位姿文件
        try:
            logging.info("Saving final pose estimation files...")
            
            # 获取所有排序后的候选位姿
            n_total_candidates = total_queries * max_candidates
            final_flat_pred_t = all_final_pred_t.reshape(total_queries, max_candidates, 3)
            final_flat_pred_R = all_final_pred_R.reshape(total_queries, max_candidates, 3, 3)
            
            # 创建flat_preds索引数组
            flat_preds = np.arange(max_candidates)[np.newaxis, :].repeat(total_queries, axis=0)
            
            # 保存多个top K结果
            top_ks = [1, 5, 10]
            
            # 使用utils.log_pose_estimate保存
            pose_files = utils.log_pose_estimate(
                args.save_dir, pose_dataset, 
                final_flat_pred_R, final_flat_pred_t, 
                flat_preds=flat_preds, top_ns=top_ks
            )
            
            logging.info(f"Final poses saved successfully")
            logging.info(f"Top K results saved: {top_ks}")
            
        except Exception as save_error:
            logging.error(f"Failed to save pose file: {save_error}")
            import traceback
            logging.error(f"Save error traceback: {traceback.format_exc()}")
        
    except Exception as e:
        logging.error(f"Failed to compute final evaluation: {e}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")


def load_batch_query_data(batch_indices, pose_dataset, args):
    """
    分批加载query数据到GPU内存
    
    Args:
        batch_indices: 当前批次的query索引列表
        pose_dataset: 数据集对象
        args: 参数对象，包含pt_base_path
    
    Returns:
        dict: {q_idx: query_data} 当前批次的预加载数据字典
    """
    batch_data = {}
    successful_loads = 0
    
    logging.info(f"开始加载批次数据: {len(batch_indices)}个query (索引 {min(batch_indices)} - {max(batch_indices)})")
    
    for q_idx in batch_indices:
        idx = pose_dataset.q_frames_idxs[q_idx]
        q_name = pose_dataset.get_basename(idx)
        q_name_clean = q_name[6:]  # 去掉 'query_' 前缀
        
        try:
            ins_pt_path = os.path.join(args.pt_base_path, "ins_pt", f"{q_name_clean}.pt")
            bbox_pt_path = os.path.join(args.pt_base_path, "bbox_pt", f"{q_name_clean}.pt")
            scores_pt_path = os.path.join(args.pt_base_path, "scores_pt", f"{q_name_clean}.pt")
            
            # 直接加载到GPU，并确保精度类型一致
            q_probs = torch.load(ins_pt_path, map_location='cuda').float()
            q_bboxs = torch.load(bbox_pt_path, map_location='cuda').float()
            
            q_weights = None
            weight_type = 'area'
            # if os.path.exists(scores_pt_path):
            #     q_scores = torch.load(scores_pt_path, map_location='cuda').float()
            #     q_weights = q_scores
            #     weight_type = 'scores'
            
            batch_data[q_idx] = {
                'q_probs': q_probs,
                'q_bboxs': q_bboxs,
                'q_weights': q_weights,
                'weight_type': weight_type
            }
            successful_loads += 1
            
        except Exception as e:
            logging.warning(f"无法加载query {q_idx} ({q_name_clean})的数据: {e}")
            batch_data[q_idx] = None
    
    # 获取GPU内存使用情况
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        cached_gb = torch.cuda.memory_reserved() / 1024**3
        memory_info = f"GPU内存: 已分配 {allocated_gb:.2f}GB, 已缓存 {cached_gb:.2f}GB"
    else:
        memory_info = "GPU不可用"
    
    logging.info(f"批次加载完成: {successful_loads}/{len(batch_indices)}个query成功加载, {memory_info}")
    return batch_data


def clear_batch_data(batch_data):
    """
    清理当前批次的GPU内存
    
    Args:
        batch_data: 当前批次的数据字典
    """
    if not batch_data:
        return
    
    cleared_count = 0
    for q_idx, query_data in batch_data.items():
        if query_data is not None:
            # 删除GPU张量引用
            if 'q_probs' in query_data and query_data['q_probs'] is not None:
                del query_data['q_probs']
            if 'q_bboxs' in query_data and query_data['q_bboxs'] is not None:
                del query_data['q_bboxs']
            if 'q_weights' in query_data and query_data['q_weights'] is not None:
                del query_data['q_weights']
            cleared_count += 1
    
    # 强制GPU内存回收
    torch.cuda.empty_cache()
    
    # 获取清理后的内存状态
    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        cached_gb = torch.cuda.memory_reserved() / 1024**3
        memory_info = f"GPU内存: 已分配 {allocated_gb:.2f}GB, 已缓存 {cached_gb:.2f}GB"
    else:
        memory_info = "GPU不可用"
    
    logging.info(f"批次内存清理完成: 清理了{cleared_count}个query的数据, {memory_info}")


def get_query_masks(dataset, transform):
    """
    获取查询图像的掩码信息
    
    Args:
        dataset: 数据集对象
        transform: 变换函数
    
    Returns:
        q_descriptors: 查询掩码列表
    """
    q_descriptors = []
    for i in range(len(dataset)):
        q_mask, _ = dataset[i]
        q_descriptors.append(q_mask)
    return q_descriptors


def sort_candidates_by_score(all_scores, all_pred_t, all_pred_R):
    """
    按照分数对所有beam的候选位姿进行排序
    类似于原版的utils.sort_preds_across_beams函数
    
    Args:
        all_scores: (n_beams, N_per_beam) Dice分数数组
        all_pred_t: (n_beams, N_per_beam, 3) 位移数组
        all_pred_R: (n_beams, N_per_beam, 3, 3) 旋转数组
    
    Returns:
        flat_pred_R: (n_total_candidates, 3, 3) 按分数排序的旋转矩阵
        flat_pred_t: (n_total_candidates, 3) 按分数排序的位移向量
        flat_preds: (n_candidates,) 排序后的索引
        sorted_scores: (n_candidates,) 排序后的分数
    """
    n_beams, N_per_beam = all_scores.shape
    
    # 将所有候选展平
    flat_scores = all_scores.flatten()  # (n_beams * N_per_beam,)
    flat_pred_t = all_pred_t.reshape(-1, 3)  # (n_beams * N_per_beam, 3)
    flat_pred_R = all_pred_R.reshape(-1, 3, 3)  # (n_beams * N_per_beam, 3, 3)
    
    # 按分数降序排序（与原版sort_preds_across_beams保持一致）
    sorted_indices = np.argsort(-flat_scores)  # 对负分数排序，等效于降序
    
    # 应用排序
    sorted_pred_t = flat_pred_t[sorted_indices]
    sorted_pred_R = flat_pred_R[sorted_indices]
    sorted_scores = flat_scores[sorted_indices]
    
    # 创建flat_preds索引数组（与原版格式兼容）
    flat_preds = sorted_indices
    
    return sorted_pred_R, sorted_pred_t, flat_preds, sorted_scores


def process_single_query(q_idx, q_name, q_mask, camera_K, w, h, 
                        initial_pred_t, initial_pred_R,
                        args, render2loc, fine_model, first_step, preloaded_query_data=None):
    """处理单个query的所有迭代"""
    
    # 设置参数
    N_steps = args.steps
    N_per_beam = args.N // args.beams
    n_beams = args.beams
    N_views = args.N
    
    # 优化4：使用预加载的数据，避免重复加载
    q_probs = None
    q_bboxs = None
    q_weights = None
    weight_type = 'area'
    
    if args.use_dice_evaluation and fine_model is not None and preloaded_query_data is not None:
        # 直接使用预加载的GPU数据
        q_probs = preloaded_query_data['q_probs']
        q_bboxs = preloaded_query_data['q_bboxs']
        q_weights = preloaded_query_data['q_weights']
        weight_type = preloaded_query_data['weight_type']
    
    # 初始化resampler
    resampler = get_protocol(args, N_per_beam, args.protocol)
    
    # 当前最佳位姿
    current_pred_t = initial_pred_t.copy()
    current_pred_R = initial_pred_R.copy()
    
    # 初始化分数记录
    query_scores = {'steps': []}
    
    # 最终的候选位姿和分数
    final_pred_t = None
    final_pred_R = None
    final_scores = {'steps': []}  # 修复：初始化为字典而不是None
    final_scores_array = None  # 初始化numpy数组格式的分数
    
    # 开始迭代
    for step in range(first_step, N_steps):
        # 只在开始时打印一次信息
        if step == first_step:
            logging.info(f'[Query {q_idx}] Starting refinement ({N_steps} steps)')
        
        resampler.init_step(step)
        center_std, angle_delta = resampler.scaler.get_noise()
        
        # 生成候选位姿并收集所有候选（传递预加载的数据）
        all_pred_t, all_pred_R, all_scores, step_errors = process_step_realtime(
            q_idx, q_name, q_mask, camera_K, w, h,
            current_pred_t, current_pred_R,
            resampler, render2loc, 
            n_beams, N_per_beam, step, args, fine_model,
            q_probs, q_bboxs, q_weights, weight_type  # 传递预加载的数据
        )
        
        # 保存最终步骤的完整候选信息
        final_pred_t = all_pred_t
        final_pred_R = all_pred_R
        final_scores_array = all_scores  # 保存numpy数组格式的分数
        
        # 对候选进行排序，选择最佳的作为下一步的输入
        sorted_pred_R, sorted_pred_t, flat_preds, sorted_scores = sort_candidates_by_score(
            all_scores, all_pred_t, all_pred_R
        )
        
        # 为下一步准备最佳候选（重构为beam格式）
        # 选择top candidates来更新current_pred
        top_candidates_per_beam = min(args.M if hasattr(args, 'M') else 2, N_per_beam)
        
        current_pred_t = np.zeros((n_beams, top_candidates_per_beam, 3))
        current_pred_R = np.zeros((n_beams, top_candidates_per_beam, 3, 3))
        
        # 将排序后的最佳候选分配到各个beam
        for beam_i in range(n_beams):
            for j in range(top_candidates_per_beam):
                idx = beam_i * top_candidates_per_beam + j
                if idx < len(sorted_pred_t):
                    current_pred_t[beam_i, j] = sorted_pred_t[idx]
                    current_pred_R[beam_i, j] = sorted_pred_R[idx]
                else:
                    # 如果候选不够，重复最佳候选
                    current_pred_t[beam_i, j] = sorted_pred_t[0]
                    current_pred_R[beam_i, j] = sorted_pred_R[0]
        
        # 记录结果
        query_scores['steps'].append(step_errors)
    
    # 完成后打印最终结果
    logging.info(f'[Query {q_idx}] Refinement completed')
    
    return query_scores, final_pred_t, final_pred_R, final_scores_array


def process_step_realtime(q_idx, q_name, q_mask, camera_K, w, h,
                         pred_t, pred_R, resampler, render2loc,
                         n_beams, N_per_beam, step, args, fine_model=None,
                         q_probs=None, q_bboxs=None, q_weights=None, weight_type='area'):
    """处理单个迭代步骤，对每个采样位姿立即计算Dice Loss分数"""
    
    # 初始化存储所有候选的数组
    all_pred_t = np.empty((n_beams, N_per_beam, 3))
    all_pred_R = np.empty((n_beams, N_per_beam, 3, 3))
    all_scores = np.empty((n_beams, N_per_beam))
    all_candidate_info = []
    
    # 优化5：预分配GPU内存用于存储渲染图像，避免动态扩展
    total_renders = n_beams * N_per_beam

    render_h, render_w = 480, 720  # 这里需要根据实际渲染分辨率调整
    # render_h, render_w = 448, 602  # 这里需要根据实际渲染分辨率调整
    # render_h, render_w = 360, 640  # 这里需要根据实际渲染分辨率调整

    all_render_images = torch.zeros((total_renders, 3, render_h, render_w), 
                                  dtype=torch.float32, device='cuda')
    all_render_info = []
    render_count = 0  # 记录实际渲染的图像数量
    
    # 对每个beam进行处理 - 只进行渲染，不计算Dice
    for beam_i in range(n_beams):
        # 正确提取beam的位姿数据
        if n_beams > 1:
            beam_pred_t = pred_t[beam_i]  # 形状: (M, 3)
            beam_pred_R = pred_R[beam_i]  # 形状: (M, 3, 3)
        else:
            beam_pred_t = pred_t[0]  # 形状: (M, 3)，取第一个beam
            beam_pred_R = pred_R[0]  # 形状: (M, 3, 3)
        
        # 生成采样位姿
        resample_result = resampler.resample(
            camera_K, q_name, beam_pred_t, beam_pred_R, q_idx=q_idx, beam_i=beam_i
        )
        
        # 处理不同协议返回值数量的差异
        if len(resample_result) == 4:
            r_names, render_ts, render_qvecs, calibr_pose = resample_result
            # 如果没有poses，从calibr_pose中提取
            poses = [pose_data[0] for pose_data in calibr_pose]
        else:
            r_names, render_ts, render_qvecs, calibr_pose, poses = resample_result
        
        # 确保有足够的采样点
        if len(poses) < N_per_beam:
            logging.warning(f"Query {q_idx}, Beam {beam_i}: Got {len(poses)} poses, expected {N_per_beam}")
            # 添加噪声的位姿来填充，避免完全重复
            original_count = len(poses)
            while len(poses) < N_per_beam:
                if original_count > 0:
                    # 选择一个已有的位姿作为基础
                    base_idx = len(poses) % original_count
                    base_pose = poses[base_idx]
                    
                    # 添加小的随机噪声避免完全重复
                    noise_scale = 0.01  # 很小的噪声
                    pose_with_noise = base_pose.copy() if base_pose is not None else None
                    if pose_with_noise is not None:
                        # 对位移添加噪声
                        pose_with_noise[0:3] += np.random.normal(0, noise_scale, 3)
                        # 对旋转添加小的噪声
                        pose_with_noise[3:7] += np.random.normal(0, noise_scale * 0.1, 4)
                        # 重新归一化四元数
                        pose_with_noise[3:7] /= np.linalg.norm(pose_with_noise[3:7])
                    
                    poses.append(pose_with_noise)
                    
                    # 也为render_ts和render_qvecs添加对应的噪声
                    if len(render_ts) > 0:
                        base_t = render_ts[base_idx % len(render_ts)]
                        noisy_t = base_t + np.random.normal(0, noise_scale, 3)
                        render_ts = np.vstack([render_ts, noisy_t.reshape(1, -1)])
                    else:
                        render_ts = np.array([[0.0, 0.0, 0.0]])
                    
                    if len(render_qvecs) > 0:
                        base_qvec = render_qvecs[base_idx % len(render_qvecs)]
                        noisy_qvec = base_qvec + np.random.normal(0, noise_scale * 0.1, 4)
                        noisy_qvec /= np.linalg.norm(noisy_qvec)
                        render_qvecs = np.vstack([render_qvecs, noisy_qvec.reshape(1, -1)])
                    else:
                        render_qvecs = np.array([[1.0, 0.0, 0.0, 0.0]])
                else:
                    # 如果没有任何位姿，创建一个默认的
                    poses.append(None)
                    render_ts = np.vstack([render_ts, np.zeros(3)] if len(render_ts) > 0 else [np.zeros(3)])
                    render_qvecs = np.vstack([render_qvecs, np.array([1,0,0,0])] if len(render_qvecs) > 0 else [np.array([1,0,0,0])])
        
        # 对每个采样位姿进行渲染（不计算Dice）
        for i in range(min(N_per_beam, len(poses))):
            pose = poses[i]
            
            # 转换位姿到WGS84坐标系
            translation_, euler_angles_ = render2loc.get_pose_w2cToWGS84_batch(np.array([pose]))
            
            # 设置渲染位姿
            render2loc.translation = [translation_[0, 0], translation_[0, 1], translation_[0, 2]]
            render2loc.euler_angles = [euler_angles_[0, 0], euler_angles_[0, 1], euler_angles_[0, 2]]
            
            # 更新渲染器位姿
            for j in range(2):
                render2loc.renderer.update_pose(render2loc.translation, render2loc.euler_angles)
            
            # 获取渲染图像 - 保持三通道彩色图
            color_image_resized = render2loc.renderer.get_color_image()
            
            # 存储候选位姿信息
            all_pred_t[beam_i, i] = render_ts[i]
            all_pred_R[beam_i, i] = qvec2rotmat(render_qvecs[i])
            
            # 优化6：直接将渲染图像存储到预分配的GPU内存中，确保精度一致
            render_image_tensor = torch.from_numpy(color_image_resized.astype(np.float32) / 255.0).permute(2, 0, 1)
            # 检查图像尺寸并调整
            actual_h, actual_w = render_image_tensor.shape[1], render_image_tensor.shape[2]
            if actual_h != render_h or actual_w != render_w:
                render_image_tensor = torch.nn.functional.interpolate(
                    render_image_tensor.unsqueeze(0), size=(render_h, render_w), 
                    mode='bilinear', align_corners=False
                ).squeeze(0)
            # 确保精度类型为float32并移到GPU
            all_render_images[render_count] = render_image_tensor.cuda().float()
            render_count += 1
            
            all_render_info.append({
                'beam_i': beam_i,
                'pose_i': i,
                'translation': render_ts[i],
                'rotation': render_qvecs[i],
                'rotation_matrix': qvec2rotmat(render_qvecs[i]),
                'pose_matrix': pose,
                'filename': f'beam{beam_i:02d}_pose{i:03d}.png' if args.save_renders else None
            })
            
            # 可选保存渲染图像
            if args.save_renders:
                render_save_dir = os.path.join(args.save_dir, 'renders')
                query_dir = os.path.join(render_save_dir, f'query_{q_idx:03d}_{q_name}')
                step_dir = os.path.join(query_dir, f'step_{step:02d}')
                os.makedirs(step_dir, exist_ok=True)
                
                filename = f'beam{beam_i:02d}_pose{i:03d}.png'
                filepath = os.path.join(step_dir, filename)
                cv2.imwrite(filepath, (color_image_resized * 255).astype(np.uint8))
    
    # 批量计算所有渲染图像的Dice分数（GPU并行，batch=B=n_beams*N_per_beam）
    if args.use_dice_evaluation and fine_model is not None and q_probs is not None and q_bboxs is not None:
        try:
            if render_count > 0:
                # 优化7：使用预分配的GPU内存，仅计算实际渲染的图像
                render_batch = all_render_images[:render_count]  # 仅使用实际渲染的部分
                with torch.no_grad():
                    batch_scores_tensor = calculate_batch_dice_score_double_improved(
                        q_probs, q_bboxs, render_batch, weight_type=weight_type, q_weights=q_weights
                    )  # [render_count]
                batch_dice_scores = batch_scores_tensor.tolist()
            else:
                batch_dice_scores = []

            # 将分数分配回对应的beam和pose位置
            score_idx = 0
            for info in all_render_info:
                beam_i = info['beam_i']
                pose_i = info['pose_i']
                dice_score = float(batch_dice_scores[score_idx]) if score_idx < len(batch_dice_scores) else 0.0
                all_scores[beam_i, pose_i] = dice_score
                info['dice_score'] = dice_score

                # 更新保存的文件名（包含分数）
                if args.save_renders and info['filename']:
                    old_filename = info['filename']
                    new_filename = f"beam{beam_i:02d}_pose{pose_i:03d}_dice{dice_score:.3f}.png"
                    render_save_dir = os.path.join(args.save_dir, 'renders')
                    query_dir = os.path.join(render_save_dir, f'query_{q_idx:03d}_{q_name}')
                    step_dir = os.path.join(query_dir, f'step_{step:02d}')
                    old_filepath = os.path.join(step_dir, old_filename)
                    new_filepath = os.path.join(step_dir, new_filename)
                    if os.path.exists(old_filepath):
                        os.rename(old_filepath, new_filepath)
                score_idx += 1

            # 将渲染信息添加到候选信息中
            all_candidate_info.extend(all_render_info)

        except Exception as e:
            logging.error(f"Query {q_idx}: Error in batch Dice Loss calculation: {e}")
            # 如果批量计算失败，使用默认分数
            for info in all_render_info:
                beam_i = info['beam_i']
                pose_i = info['pose_i']
                all_scores[beam_i, pose_i] = 0.0
                info['dice_score'] = 0.0
            all_candidate_info.extend(all_render_info)
    else:
        # 如果没有启用Dice评估，使用默认分数
        for info in all_render_info:
            beam_i = info['beam_i']
            pose_i = info['pose_i']
            all_scores[beam_i, pose_i] = 0.5
            info['dice_score'] = 0.5
        all_candidate_info.extend(all_render_info)

    # 保存最佳候选的渲染结果
    if args.save_renders and len(all_candidate_info) > 0:
        # 找到Dice分数最高的候选
        best_score = np.max(all_scores)
        best_beam_i, best_pose_i = np.unravel_index(np.argmax(all_scores), all_scores.shape)
        best_candidate_idx = best_beam_i * N_per_beam + best_pose_i
        
        if best_candidate_idx < len(all_candidate_info):
            best_candidate = all_candidate_info[best_candidate_idx]
            
            # 重新渲染最佳候选位姿
            translation_, euler_angles_ = render2loc.get_pose_w2cToWGS84_batch(np.array([best_candidate['pose_matrix']]))
            render2loc.translation = [translation_[0, 0], translation_[0, 1], translation_[0, 2]]
            render2loc.euler_angles = [euler_angles_[0, 0], euler_angles_[0, 1], euler_angles_[0, 2]]
            for i in range(3):
                render2loc.renderer.update_pose(render2loc.translation, render2loc.euler_angles)
            
            # 保存最佳结果到query目录
            render_save_dir = os.path.join(args.save_dir, 'renders')
            query_dir = os.path.join(render_save_dir, f'query_{q_idx:03d}_{q_name}')
            os.makedirs(query_dir, exist_ok=True)
            
            best_filename = f'step_{step:02d}_BEST_dice{best_score:.3f}.png'
            best_filepath = os.path.join(query_dir, best_filename)
            render2loc.renderer.save_color_image(best_filepath)
    
    # 计算误差（这里简化，实际应该与ground truth比较）
    step_errors = {
        't': 0.0,  # 这里需要实际的ground truth来计算
        'R': 0.0
    }
    
    return all_pred_t, all_pred_R, all_scores, step_errors


if __name__ == "__main__":
    args = parse_args()
    
    # 添加 Dice Loss 评估相关参数
    if not hasattr(args, 'use_dice_evaluation'):
        args.use_dice_evaluation = True  # 默认启用 Dice Loss 评估
    
    # pt_base_path 现在从命令行参数获取，无需硬编码
    logging.info(f"Using pt_base_path: {args.pt_base_path}")
    
    # 优化8：GPU性能设置
    torch.backends.cudnn.benchmark = True  # 启用cuDNN自动调优
    torch.cuda.empty_cache()  # 清理GPU缓存
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)  # 预留部分GPU内存
    
    # 启用渲染结果保存（可选）
    args.save_renders = False  # 设置为True以保存渲染图像

    main(args)
