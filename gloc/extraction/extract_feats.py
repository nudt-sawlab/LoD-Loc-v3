import logging
import os
import torch
from tqdm import tqdm
from os.path import join
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Subset


def extract_features(model, model_name, pose_dataset, res, bs=32, check_cache=True):
    pd = pose_dataset
    DS = pd.name

    q_cache_file = join('descr_cache',f'{DS}_{model_name}_{res}_q_descriptors.pth')
    db_cache_file = join('descr_cache', f'{DS}_{model_name}_{res}_db_descriptors.pth')
    if check_cache:
        if (os.path.isfile(db_cache_file) and os.path.isfile(q_cache_file)):

            logging.info(f"Loading {db_cache_file}")
            db_descriptors = torch.load(db_cache_file)
            q_descriptors = torch.load(q_cache_file)
        
            return db_descriptors, q_descriptors

    model = model.eval()
        
    queries_subset_ds = Subset(pd, pd.q_frames_idxs)
    database_subset_ds = Subset(pd, pd.db_frames_idxs)

    db_descriptors = get_query_features(model, database_subset_ds,  bs)
    q_descriptors = get_query_features(model, queries_subset_ds, bs)
    db_descriptors = np.vstack(db_descriptors)
    q_descriptors = np.vstack(q_descriptors)

    if check_cache:
        os.makedirs('descr_cache', exist_ok=True)
        torch.save(db_descriptors, db_cache_file)
        torch.save(q_descriptors, q_cache_file)

    return db_descriptors, q_descriptors


def get_query_features(model, dataset, feat_level, bs=1):
    """
    Separate function for the queries as they might have different
    resolutions; thus it does not use a matrix to store descriptors
    but a list of arrays
    """
    model = model.eval()
    # bs = 1 as resolution might differ
    dataloader = DataLoader(dataset=dataset, num_workers=8, batch_size=bs)

    iterator = tqdm(dataloader, ncols=100)
    descriptors = []
    with torch.no_grad():
        for images in iterator:
            images = images['im'].cuda()    

            descr = model(images)   
            descr = descr[feat_level].cpu().numpy()  # 取Path1(即分辨率最高的一級)
            descriptors.append(descr)

    return descriptors

def get_query_mask(dataset, bs=1):
    """
    Separate function for the queries as they might have different
    resolutions; thus it does not use a matrix to store descriptors
    but a list of arrays
    单独为查询集设计的函数，因为查询图片可能分辨率不同，
    所以不使用统一的矩阵存储描述符，而是用数组列表。
    """
    # bs = 1 as resolution might differ
    # bs = 1，因为分辨率可能不同，不能批量堆叠，但是可以多线程加载图片
    dataloader = DataLoader(dataset=dataset, num_workers=8, batch_size=bs)

    images_list = []
    for batch in tqdm(dataloader, ncols=100):
        # (Pdb) batch
        # {'im_ref': Image(id=tensor([0]), qvec=tensor([[ 0.5483,  0.8318, -0.0731,  0.0463]], dtype=torch.float64), tvec=tensor([[-207.7588,  -19.1323,  274.6899]], dtype=torch.float64), camera_id=tensor([1563]), name=('query/DJI_20231018092903_0016_D.JPG',), xys={}, point3D_ids={}), 'im': tensor([[[[0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0039, 0.0039],
        #   [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        #   [0.0000, 0.0000, 0.0000,  ..., 0.0000, 0.0000, 0.0000],
        #   ...,
        #   [0.0000, 0.0000, 0.0000,  ..., 0.0078, 0.0078, 0.0078],
        #   [0.0000, 0.0000, 0.0000,  ..., 0.0039, 0.0039, 0.0039],
        #   [0.0000, 0.0000, 0.0000,  ..., 0.0039, 0.0039, 0.0039]]]])}

        # - 'im_ref' ：原始图片的元数据对象（包含文件名、位姿等）
        # - 'im' ：经过 transform 处理后的图片张量（Tensor）  
        images = batch['im'].cuda()  # 提取图像并将其移动到 GPU（如果需要）
        if len(images.shape) == 4:# 如果是4维（批量、通道、高、宽），只取第一个通道
            images = images[:,0, :, :]  # 选择第一个通道
        # # 添加二值化处理
        # images = (images > 0.5).float()  # 将大于0.5的值设为1，其他设为0
        
        images_list.append(images.squeeze())# 压缩多余维度后加入列表

    return images_list

def get_candidates_features(model, dataset, descr_dim,feat_level, bs=32):
    dl = DataLoader(dataset=dataset, num_workers=8, batch_size=bs)

    len_ds = len(dataset)
    descriptors = np.empty((len_ds, *descr_dim), dtype=np.float32)

    with torch.no_grad():
        for i, images in enumerate(tqdm(dl, ncols=100)):
            descr = model(images.cuda())
            
            descr = descr[feat_level].cpu().numpy()
            descriptors[i*bs:(i+1)*bs] = descr

    return descriptors

def get_candidates_mask(dataset, descr_dim, bs=32):
    dl = DataLoader(dataset=dataset, num_workers=8, batch_size=bs)

    len_ds = len(dataset)
    # descr_dim = (360, 640)  # Japan_02_one_third
    # descr_dim = (1080, 1920)  # Japan_02
    # descr_dim = (448, 602)  # 改 inTraj outTraj
    descr_dim = (480, 720)  # 改 Swiss
    # descr_dim = (480, 270)  # 改 Video
    
    # descriptors = np.empty((len_ds, *descr_dim), dtype=np.float32)
    # 修改为支持RGB图像的维度：(len_ds, 3, height, width)
    descriptors = np.empty((len_ds, 3, *descr_dim), dtype=np.float32)

    with torch.no_grad():
        for i, images in enumerate(tqdm(dl, ncols=100)):
            # breakpoint()
            descriptors[i*bs:(i+1)*bs] = images.squeeze()

    return descriptors
