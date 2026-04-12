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

import matplotlib
matplotlib.use('Agg')


def calculate_batch_dice_score_double_improved(q_probs, q_bboxs, render_images_batch, weight_type='area', q_weights=None, threshold=0.1):

    assert isinstance(q_probs, torch.Tensor) and q_probs.is_cuda
    assert isinstance(q_bboxs, torch.Tensor) and q_bboxs.is_cuda
    assert isinstance(render_images_batch, torch.Tensor) and render_images_batch.is_cuda


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
    

    q_sums = q_probs.sum(dim=(1, 2))  # [C]

    B = render_images_batch.shape[0]
    out_scores = torch.zeros(B, device=q_probs.device, dtype=torch.float32)

    eps = 1e-6



    render_2d_batch = (render_images_batch[:, 0] * 65536 + render_images_batch[:, 1] * 256 + render_images_batch[:, 2]).to(torch.int32)
    
    for b in range(B):
        render_2d = render_2d_batch[b]

        unique_colors = torch.unique(render_2d)
        unique_colors = unique_colors[unique_colors > 0]
        if unique_colors.numel() == 0:
            out_scores[b] = 0.0
            continue


        color_vals = unique_colors.view(-1, 1, 1)
        color_masks = (render_2d.unsqueeze(0) == color_vals).to(torch.float32)


        # intersections: [K, C] = sum_hw(color_masks[K,H,W] * q_probs[C,H,W])
        intersections = torch.einsum('khw,chw->kc', color_masks, q_probs)
        r_sums = color_masks.sum(dim=(1, 2))  # [K]

        denom = q_sums.unsqueeze(0) + r_sums.unsqueeze(1) + eps  # [K, C]
        dice_kc = (2.0 * intersections + eps) / denom


        q2r_per_c = dice_kc.max(dim=0).values  # [C]
        q2r_score = (q2r_per_c * weights).sum()

        out_scores[b] = q2r_score
        

    return out_scores


def main(args):

    if not hasattr(args, 'save_renders'):
        args.save_renders = False
    
    commons.make_deterministic(args.seed)
    commons.setup_logging(args.save_dir, console="info")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")
    

    if args.save_renders:
        render_save_dir = os.path.join(args.save_dir, 'renders')
        os.makedirs(render_save_dir, exist_ok=True)
        logging.info(f"Rendered images will be saved to {render_save_dir}")
    
    paths_conf = get_path_conf(args.colmap_res, args.mesh)
    exp_config = get_config(args.name)
    

    if exp_config and not isinstance(exp_config, type(NotImplementedError)):

        main_config = exp_config[0]
        

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
    

    args.N = int(args.N) if hasattr(args, 'N') else 52
    args.beams = int(args.beams) if hasattr(args, 'beams') else 2
    args.M = int(args.M) if hasattr(args, 'M') else 2
    args.steps = int(args.steps) if hasattr(args, 'steps') else 1
    

    render2loc = RealTime_render(args.render_config)
    

    DS = args.name
    res = args.res
    colmap_dir = paths_conf[DS]['colmap']
    if isinstance(colmap_dir, list):
        colmap_dir = colmap_dir[0]
    transform = get_transform(args, colmap_dir)
    pose_dataset = get_dataset(DS, paths_conf[DS], transform)
    

    queries_subset = Subset(pose_dataset, pose_dataset.q_frames_idxs)
    q_descriptors = get_query_masks(queries_subset, transform)


    fine_model = None
    if args.use_dice_evaluation:

        fine_model = True
        logging.info("Dice evaluation enabled with improved integrated function")
    

    first_step, all_pred_t, all_pred_R, scores = initialization.init_refinement(args, pose_dataset)
    

    if all_pred_t is None or all_pred_R is None:
        logging.error("Failed to get initial predictions")
        return
    

    final_scores = scores
    total_queries = len(pose_dataset.q_frames_idxs)
    

    batch_size = 100
    total_batches = (total_queries + batch_size - 1) // batch_size
    logging.info(f"将分{total_batches}个批次处理{total_queries}个query，每批最多{batch_size}个")
    

    max_candidates = args.N
    all_final_pred_t = np.zeros((total_queries, max_candidates, 3))
    all_final_pred_R = np.zeros((total_queries, max_candidates, 3, 3))
    all_final_scores = np.zeros((total_queries, max_candidates))
    

    for batch_idx in range(total_batches):

        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, total_queries)
        batch_indices = list(range(batch_start, batch_end))
        
        processed_queries = batch_start
        remaining_queries = total_queries - processed_queries
        progress_percent = (processed_queries / total_queries) * 100
        logging.info(f"开始处理批次 {batch_idx + 1}/{total_batches}: query {batch_start}-{batch_end-1} ({len(batch_indices)}个)")
        logging.info(f"总体进度: {processed_queries}/{total_queries} ({progress_percent:.1f}%), 剩余 {remaining_queries} 个query")
        

        batch_data = {}
        if args.use_dice_evaluation and fine_model:
            batch_data = load_batch_query_data(batch_indices, pose_dataset, args)
        

        for q_idx in tqdm(batch_indices, desc=f"Batch {batch_idx+1}/{total_batches}"):

            idx = pose_dataset.q_frames_idxs[q_idx]
            q_name = pose_dataset.get_basename(idx)
            q_name_clean = q_name[6:]
            q_key_name = os.path.splitext(pose_dataset.images[idx].name)[0]
            

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

            query_pred_t = all_pred_t[q_idx]
            query_pred_R = all_pred_R[q_idx]
            

            preloaded_query_data = batch_data.get(q_idx, None)
            

            query_scores, final_pred_t, final_pred_R, query_scores_array = process_single_query(
                q_idx, q_name_clean, q_mask, camera_K, w, h,
                query_pred_t, query_pred_R,
                args, render2loc, fine_model,
                first_step, preloaded_query_data
            )
            

            if final_pred_t is not None and final_pred_R is not None and query_scores_array is not None:

                sorted_pred_R, sorted_pred_t, flat_preds, sorted_scores = sort_candidates_by_score(
                    query_scores_array, final_pred_t, final_pred_R
                )
                

                n_candidates = min(len(sorted_pred_t), max_candidates)
                all_final_pred_t[q_idx, :n_candidates] = sorted_pred_t[:n_candidates]
                all_final_pred_R[q_idx, :n_candidates] = sorted_pred_R[:n_candidates]
                all_final_scores[q_idx, :n_candidates] = sorted_scores[:n_candidates]
                

                if n_candidates < max_candidates:
                    for i in range(n_candidates, max_candidates):
                        all_final_pred_t[q_idx, i] = sorted_pred_t[0]
                        all_final_pred_R[q_idx, i] = sorted_pred_R[0]
                        all_final_scores[q_idx, i] = sorted_scores[0]
            else:

                logging.warning(f"No final poses for query {q_idx}, using initial poses")
                for i in range(max_candidates):
                    all_final_pred_t[q_idx, i] = all_pred_t[q_idx, 0, 0]
                    all_final_pred_R[q_idx, i] = all_pred_R[q_idx, 0, 0]
                    all_final_scores[q_idx, i] = 0.0
            

            final_scores = utils.update_scores(final_scores, query_scores)
        

        if batch_data:
            clear_batch_data(batch_data)
            logging.info(f"批次 {batch_idx + 1}/{total_batches} 处理完成并清理内存")
    

    logging.info("="*50)
    logging.info("COMPUTING FINAL EVALUATION FOR ALL QUERIES")
    logging.info("="*50)
    
    try:

        all_true_t, all_true_R = pose_dataset.get_q_poses()
        

        final_pred_t = all_final_pred_t[:, 0, :]  # (n_queries, 3)
        final_pred_R = all_final_pred_R[:, 0, :, :]  # (n_queries, 3, 3)
        

        final_pred_t = final_pred_t.reshape(total_queries, 1, 3)  # (n_queries, 1, 3)
        final_pred_R = final_pred_R.reshape(total_queries, 1, 3, 3)  # (n_queries, 1, 3, 3)
        

        errors_t, errors_R = utils.get_all_errors_first_estimate(
            all_true_t, all_true_R, final_pred_t, final_pred_R
        )
        

        all_errors_t = errors_t.reshape(total_queries, 1)  # (n_queries, 1)
        all_errors_R = errors_R.reshape(total_queries, 1)  # (n_queries, 1)
        

        result_str, results = utils.eval_poses_top_n(
            all_errors_t, all_errors_R, descr=f'step {args.steps}'
        )
        
        logging.info("FINAL EVALUATION RESULTS:")
        logging.info(result_str)
        

        try:
            logging.info("Saving final pose estimation files...")
            

            n_total_candidates = total_queries * max_candidates
            final_flat_pred_t = all_final_pred_t.reshape(total_queries, max_candidates, 3)
            final_flat_pred_R = all_final_pred_R.reshape(total_queries, max_candidates, 3, 3)
            

            flat_preds = np.arange(max_candidates)[np.newaxis, :].repeat(total_queries, axis=0)
            

            top_ks = [1, 5, 10]
            

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
    batch_data = {}
    successful_loads = 0
    
    logging.info(f"开始加载批次数据: {len(batch_indices)}个query (索引 {min(batch_indices)} - {max(batch_indices)})")
    
    for q_idx in batch_indices:
        idx = pose_dataset.q_frames_idxs[q_idx]
        q_name = pose_dataset.get_basename(idx)
        q_name_clean = q_name[6:]
        
        try:
            ins_pt_path = os.path.join(args.pt_base_path, "ins_pt", f"{q_name_clean}.pt")
            bbox_pt_path = os.path.join(args.pt_base_path, "bbox_pt", f"{q_name_clean}.pt")
            scores_pt_path = os.path.join(args.pt_base_path, "scores_pt", f"{q_name_clean}.pt")
            

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
    

    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        cached_gb = torch.cuda.memory_reserved() / 1024**3
        memory_info = f"GPU内存: 已分配 {allocated_gb:.2f}GB, 已缓存 {cached_gb:.2f}GB"
    else:
        memory_info = "GPU不可用"
    
    logging.info(f"批次加载完成: {successful_loads}/{len(batch_indices)}个query成功加载, {memory_info}")
    return batch_data


def clear_batch_data(batch_data):
    if not batch_data:
        return
    
    cleared_count = 0
    for q_idx, query_data in batch_data.items():
        if query_data is not None:

            if 'q_probs' in query_data and query_data['q_probs'] is not None:
                del query_data['q_probs']
            if 'q_bboxs' in query_data and query_data['q_bboxs'] is not None:
                del query_data['q_bboxs']
            if 'q_weights' in query_data and query_data['q_weights'] is not None:
                del query_data['q_weights']
            cleared_count += 1
    

    torch.cuda.empty_cache()
    

    if torch.cuda.is_available():
        allocated_gb = torch.cuda.memory_allocated() / 1024**3
        cached_gb = torch.cuda.memory_reserved() / 1024**3
        memory_info = f"GPU内存: 已分配 {allocated_gb:.2f}GB, 已缓存 {cached_gb:.2f}GB"
    else:
        memory_info = "GPU不可用"
    
    logging.info(f"批次内存清理完成: 清理了{cleared_count}个query的数据, {memory_info}")


def get_query_masks(dataset, transform):
    q_descriptors = []
    for i in range(len(dataset)):
        q_mask, _ = dataset[i]
        q_descriptors.append(q_mask)
    return q_descriptors


def sort_candidates_by_score(all_scores, all_pred_t, all_pred_R):
    n_beams, N_per_beam = all_scores.shape
    

    flat_scores = all_scores.flatten()  # (n_beams * N_per_beam,)
    flat_pred_t = all_pred_t.reshape(-1, 3)  # (n_beams * N_per_beam, 3)
    flat_pred_R = all_pred_R.reshape(-1, 3, 3)  # (n_beams * N_per_beam, 3, 3)
    

    sorted_indices = np.argsort(-flat_scores)
    

    sorted_pred_t = flat_pred_t[sorted_indices]
    sorted_pred_R = flat_pred_R[sorted_indices]
    sorted_scores = flat_scores[sorted_indices]
    

    flat_preds = sorted_indices
    
    return sorted_pred_R, sorted_pred_t, flat_preds, sorted_scores


def process_single_query(q_idx, q_name, q_mask, camera_K, w, h, 
                        initial_pred_t, initial_pred_R,
                        args, render2loc, fine_model, first_step, preloaded_query_data=None):
    

    N_steps = args.steps
    N_per_beam = args.N // args.beams
    n_beams = args.beams
    N_views = args.N
    

    q_probs = None
    q_bboxs = None
    q_weights = None
    weight_type = 'area'
    
    if args.use_dice_evaluation and fine_model is not None and preloaded_query_data is not None:

        q_probs = preloaded_query_data['q_probs']
        q_bboxs = preloaded_query_data['q_bboxs']
        q_weights = preloaded_query_data['q_weights']
        weight_type = preloaded_query_data['weight_type']
    

    resampler = get_protocol(args, N_per_beam, args.protocol)
    

    current_pred_t = initial_pred_t.copy()
    current_pred_R = initial_pred_R.copy()
    

    query_scores = {'steps': []}
    

    final_pred_t = None
    final_pred_R = None
    final_scores = {'steps': []}
    final_scores_array = None
    

    for step in range(first_step, N_steps):

        if step == first_step:
            logging.info(f'[Query {q_idx}] Starting refinement ({N_steps} steps)')
        
        resampler.init_step(step)
        center_std, angle_delta = resampler.scaler.get_noise()
        

        all_pred_t, all_pred_R, all_scores, step_errors = process_step_realtime(
            q_idx, q_name, q_mask, camera_K, w, h,
            current_pred_t, current_pred_R,
            resampler, render2loc, 
            n_beams, N_per_beam, step, args, fine_model,
            q_probs, q_bboxs, q_weights, weight_type
        )
        

        final_pred_t = all_pred_t
        final_pred_R = all_pred_R
        final_scores_array = all_scores
        

        sorted_pred_R, sorted_pred_t, flat_preds, sorted_scores = sort_candidates_by_score(
            all_scores, all_pred_t, all_pred_R
        )
        


        top_candidates_per_beam = min(args.M if hasattr(args, 'M') else 2, N_per_beam)
        
        current_pred_t = np.zeros((n_beams, top_candidates_per_beam, 3))
        current_pred_R = np.zeros((n_beams, top_candidates_per_beam, 3, 3))
        

        for beam_i in range(n_beams):
            for j in range(top_candidates_per_beam):
                idx = beam_i * top_candidates_per_beam + j
                if idx < len(sorted_pred_t):
                    current_pred_t[beam_i, j] = sorted_pred_t[idx]
                    current_pred_R[beam_i, j] = sorted_pred_R[idx]
                else:

                    current_pred_t[beam_i, j] = sorted_pred_t[0]
                    current_pred_R[beam_i, j] = sorted_pred_R[0]
        

        query_scores['steps'].append(step_errors)
    

    logging.info(f'[Query {q_idx}] Refinement completed')
    
    return query_scores, final_pred_t, final_pred_R, final_scores_array


def process_step_realtime(q_idx, q_name, q_mask, camera_K, w, h,
                         pred_t, pred_R, resampler, render2loc,
                         n_beams, N_per_beam, step, args, fine_model=None,
                         q_probs=None, q_bboxs=None, q_weights=None, weight_type='area'):
    

    all_pred_t = np.empty((n_beams, N_per_beam, 3))
    all_pred_R = np.empty((n_beams, N_per_beam, 3, 3))
    all_scores = np.empty((n_beams, N_per_beam))
    all_candidate_info = []
    

    total_renders = n_beams * N_per_beam

    render_h, render_w = 480, 720



    all_render_images = torch.zeros((total_renders, 3, render_h, render_w), 
                                  dtype=torch.float32, device='cuda')
    all_render_info = []
    render_count = 0
    

    for beam_i in range(n_beams):

        if n_beams > 1:
            beam_pred_t = pred_t[beam_i]
            beam_pred_R = pred_R[beam_i]
        else:
            beam_pred_t = pred_t[0]
            beam_pred_R = pred_R[0]
        

        resample_result = resampler.resample(
            camera_K, q_name, beam_pred_t, beam_pred_R, q_idx=q_idx, beam_i=beam_i
        )
        

        if len(resample_result) == 4:
            r_names, render_ts, render_qvecs, calibr_pose = resample_result

            poses = [pose_data[0] for pose_data in calibr_pose]
        else:
            r_names, render_ts, render_qvecs, calibr_pose, poses = resample_result
        

        if len(poses) < N_per_beam:
            logging.warning(f"Query {q_idx}, Beam {beam_i}: Got {len(poses)} poses, expected {N_per_beam}")

            original_count = len(poses)
            while len(poses) < N_per_beam:
                if original_count > 0:

                    base_idx = len(poses) % original_count
                    base_pose = poses[base_idx]
                    

                    noise_scale = 0.01
                    pose_with_noise = base_pose.copy() if base_pose is not None else None
                    if pose_with_noise is not None:

                        pose_with_noise[0:3] += np.random.normal(0, noise_scale, 3)

                        pose_with_noise[3:7] += np.random.normal(0, noise_scale * 0.1, 4)

                        pose_with_noise[3:7] /= np.linalg.norm(pose_with_noise[3:7])
                    
                    poses.append(pose_with_noise)
                    

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

                    poses.append(None)
                    render_ts = np.vstack([render_ts, np.zeros(3)] if len(render_ts) > 0 else [np.zeros(3)])
                    render_qvecs = np.vstack([render_qvecs, np.array([1,0,0,0])] if len(render_qvecs) > 0 else [np.array([1,0,0,0])])
        

        for i in range(min(N_per_beam, len(poses))):
            pose = poses[i]
            

            translation_, euler_angles_ = render2loc.get_pose_w2cToWGS84_batch(np.array([pose]))
            

            render2loc.translation = [translation_[0, 0], translation_[0, 1], translation_[0, 2]]
            render2loc.euler_angles = [euler_angles_[0, 0], euler_angles_[0, 1], euler_angles_[0, 2]]
            

            for j in range(2):
                render2loc.renderer.update_pose(render2loc.translation, render2loc.euler_angles)
            

            color_image_resized = render2loc.renderer.get_color_image()
            

            all_pred_t[beam_i, i] = render_ts[i]
            all_pred_R[beam_i, i] = qvec2rotmat(render_qvecs[i])
            

            render_image_tensor = torch.from_numpy(color_image_resized.astype(np.float32) / 255.0).permute(2, 0, 1)

            actual_h, actual_w = render_image_tensor.shape[1], render_image_tensor.shape[2]
            if actual_h != render_h or actual_w != render_w:
                render_image_tensor = torch.nn.functional.interpolate(
                    render_image_tensor.unsqueeze(0), size=(render_h, render_w), 
                    mode='bilinear', align_corners=False
                ).squeeze(0)

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
            

            if args.save_renders:
                render_save_dir = os.path.join(args.save_dir, 'renders')
                query_dir = os.path.join(render_save_dir, f'query_{q_idx:03d}_{q_name}')
                step_dir = os.path.join(query_dir, f'step_{step:02d}')
                os.makedirs(step_dir, exist_ok=True)
                
                filename = f'beam{beam_i:02d}_pose{i:03d}.png'
                filepath = os.path.join(step_dir, filename)
                cv2.imwrite(filepath, (color_image_resized * 255).astype(np.uint8))
    

    if args.use_dice_evaluation and fine_model is not None and q_probs is not None and q_bboxs is not None:
        try:
            if render_count > 0:

                render_batch = all_render_images[:render_count]
                with torch.no_grad():
                    batch_scores_tensor = calculate_batch_dice_score_double_improved(
                        q_probs, q_bboxs, render_batch, weight_type=weight_type, q_weights=q_weights
                    )  # [render_count]
                batch_dice_scores = batch_scores_tensor.tolist()
            else:
                batch_dice_scores = []


            score_idx = 0
            for info in all_render_info:
                beam_i = info['beam_i']
                pose_i = info['pose_i']
                dice_score = float(batch_dice_scores[score_idx]) if score_idx < len(batch_dice_scores) else 0.0
                all_scores[beam_i, pose_i] = dice_score
                info['dice_score'] = dice_score


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


            all_candidate_info.extend(all_render_info)

        except Exception as e:
            logging.error(f"Query {q_idx}: Error in batch Dice Loss calculation: {e}")

            for info in all_render_info:
                beam_i = info['beam_i']
                pose_i = info['pose_i']
                all_scores[beam_i, pose_i] = 0.0
                info['dice_score'] = 0.0
            all_candidate_info.extend(all_render_info)
    else:

        for info in all_render_info:
            beam_i = info['beam_i']
            pose_i = info['pose_i']
            all_scores[beam_i, pose_i] = 0.5
            info['dice_score'] = 0.5
        all_candidate_info.extend(all_render_info)


    if args.save_renders and len(all_candidate_info) > 0:

        best_score = np.max(all_scores)
        best_beam_i, best_pose_i = np.unravel_index(np.argmax(all_scores), all_scores.shape)
        best_candidate_idx = best_beam_i * N_per_beam + best_pose_i
        
        if best_candidate_idx < len(all_candidate_info):
            best_candidate = all_candidate_info[best_candidate_idx]
            

            translation_, euler_angles_ = render2loc.get_pose_w2cToWGS84_batch(np.array([best_candidate['pose_matrix']]))
            render2loc.translation = [translation_[0, 0], translation_[0, 1], translation_[0, 2]]
            render2loc.euler_angles = [euler_angles_[0, 0], euler_angles_[0, 1], euler_angles_[0, 2]]
            for i in range(3):
                render2loc.renderer.update_pose(render2loc.translation, render2loc.euler_angles)
            

            render_save_dir = os.path.join(args.save_dir, 'renders')
            query_dir = os.path.join(render_save_dir, f'query_{q_idx:03d}_{q_name}')
            os.makedirs(query_dir, exist_ok=True)
            
            best_filename = f'step_{step:02d}_BEST_dice{best_score:.3f}.png'
            best_filepath = os.path.join(query_dir, best_filename)
            render2loc.renderer.save_color_image(best_filepath)
    

    step_errors = {
        't': 0.0,
        'R': 0.0
    }
    
    return all_pred_t, all_pred_R, all_scores, step_errors


if __name__ == "__main__":
    args = parse_args()
    

    if not hasattr(args, 'use_dice_evaluation'):
        args.use_dice_evaluation = True
    

    logging.info(f"Using pt_base_path: {args.pt_base_path}")
    

    torch.backends.cudnn.benchmark = True
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.9)
    

    args.save_renders = False

    main(args)
