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
            descr = descr[feat_level].cpu().numpy()
            descriptors.append(descr)

    return descriptors

def get_query_mask(dataset, bs=1):
    # bs = 1 as resolution might differ

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



        images = batch['im'].cuda()
        if len(images.shape) == 4:
            images = images[:,0, :, :]


        
        images_list.append(images.squeeze())

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

    descr_dim = (480, 720)

    
    # descriptors = np.empty((len_ds, *descr_dim), dtype=np.float32)

    descriptors = np.empty((len_ds, 3, *descr_dim), dtype=np.float32)

    with torch.no_grad():
        for i, images in enumerate(tqdm(dl, ncols=100)):
            # breakpoint()
            descriptors[i*bs:(i+1)*bs] = images.squeeze()

    return descriptors
