# type: ignore
import os  # 导入os模块
from os.path import join, dirname  # 导入路径拼接和获取父目录的函数
import torch  # 导入PyTorch
import logging  # 导入日志模块

from gloc import extraction  # 导入gloc/extraction.py
from gloc.utils import utils  # 导入gloc/utils/utils.py


def init_refinement(args, pose_dataset):  # 定义初始化精化流程的函数
    first_step = 0  # 初始化first_step为0
    scores = {}  # 初始化分数字典

    if (args.pose_prior is None) and (args.resume_step is None):  # 如果没有先验位姿且没有恢复步骤
        # 如果没有位姿先验，则用检索初始化
        all_pred_t, all_pred_R = extraction.get_retrieval_predictions(args.retr_model, args.res, pose_dataset, args.beams*args.M)# args.beams*args.M 2*2=4
    elif args.pose_prior is not None:  # 如果有位姿先验
        assert os.path.isfile(args.pose_prior), f'{args.pose_prior} does not exist as a file'  # 检查先验文件是否存在

        logging.info(f'Loading pose prior from {args.pose_prior}')  # 日志输出加载先验
        # (Pdb) all_pred_t[0]
        # array([[-208.967456,  -23.53692 ,  273.411464],
        #     [-208.967456,  -23.53692 ,  273.411464],
        #     [-208.967456,  -23.53692 ,  273.411464],
        #     [-208.967456,  -23.53692 ,  273.411464]])
        # (Pdb) all_pred_R[0]
        # array([[[ 0.98685079, -0.16160218, -0.00320136],
        #         [-0.06664153, -0.38875296, -0.91892807],
        #         [ 0.1472563 ,  0.90705823, -0.39441058]],

        #     [[ 0.98685079, -0.16160218, -0.00320136],
        #         [-0.06664153, -0.38875296, -0.91892807],
        #         [ 0.1472563 ,  0.90705823, -0.39441058]],

        #     [[ 0.98685079, -0.16160218, -0.00320136],
        #         [-0.06664153, -0.38875296, -0.91892807],
        #         [ 0.1472563 ,  0.90705823, -0.39441058]],

        #     [[ 0.98685079, -0.16160218, -0.00320136],
        #         [-0.06664153, -0.38875296, -0.91892807],
        #         [ 0.1472563 ,  0.90705823, -0.39441058]]])
        # 每张图生成4个相同的先验位姿 beams×M
        # 输出 all_pred_t 和 all_pred_R ，分别为 shape=(n_q, M, 3) 和 (n_q, M, 3, 3) 的平移和旋转矩阵数组。
        all_pred_t, all_pred_R = utils.load_pose_prior(args.pose_prior, pose_dataset, args.beams*args.M)  # 加载先验位姿

    if args.resume_step is None:  # 如果没有恢复步骤
        # shape=(n_q, 1, 3) 和 (n_q, 1, 3, 3)
        all_true_t, all_true_R = pose_dataset.get_q_poses()  # 获取所有查询图的真实位姿
        errors_t, errors_R = utils.get_all_errors_first_estimate(all_true_t, all_true_R, all_pred_t, all_pred_R)  # 计算初始估计误差
        # breakpoint()
        # out_str: 各个阈值下结果
        # out_vals: 各个阈值下的比率
        out_str, out_vals = utils.eval_poses(errors_t, errors_R, descr='Retrieval first estimate')  # 评估初始位姿
        scores['baseline'] = out_vals  # 保存基线分数
        logging.info(out_str)  # 日志输出评估结果

        # 将预测结果reshape为 (NQ, beams, M, 3)/(NQ, beams, M, 3, 3) 方便后续处理
        all_pred_t = utils.reshape_preds_per_beam(args.beams, args.M, all_pred_t)
        all_pred_R = utils.reshape_preds_per_beam(args.beams, args.M, all_pred_R)
    else:
        all_pred_t, all_pred_R = None, None  # 如果有恢复步骤，则预测为None

    scores['steps'] = []  # 初始化步骤分数列表
    if args.first_step is not None:  # 如果指定了first_step
        first_step = args.first_step
        if args.resume_step is not None:  # 如果有恢复步骤
            score_path = join(dirname(dirname(args.resume_step)), 'scores.pth')  # 构造分数文件路径
            if os.path.isfile(score_path):  # 如果分数文件存在
                scores = torch.load(score_path)  # 加载分数
                if len(scores['steps']) > first_step:  # 如果分数步数多于first_step
                    logging.info(f"Cutting score file to {first_step},  from {len(scores['steps'])}")
                    scores['steps'] = scores['steps'][:first_step]  # 截断分数到first_step

    return first_step, all_pred_t, all_pred_R, scores  # 返回初始化结果 0，all_pred_t, all_pred_R,
    # (Pdb) scores
    # {'baseline': array([0.01231752, 0.11678832, 0.29881387, 0.51140511, 0.97262774]), 'steps': []}