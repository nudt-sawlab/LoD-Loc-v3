from os.path import join
from PIL import Image
import os
import torch.utils.data as data
from tqdm import tqdm

from gloc.datasets import RenderedImagesDataset


class ImListDataset(data.Dataset):
    def __init__(self, path_list, transform=None):
        self.path_list = path_list
        self.transform = transform
        
    def __getitem__(self, idx):
        # im = Image.open(self.path_list[idx]).convert('L')#.convert('RGB')
        im = Image.open(self.path_list[idx])#.convert('RGB')   
        if self.transform:
            im = self.transform(im)
        return im
    
    def __len__(self):
        return len(self.path_list)


def find_candidates_paths(pose_dataset, n_beams, render_dir):
    candidates_pathlist = []
    for q_idx in tqdm(range(len(pose_dataset.q_frames_idxs)), ncols=100):
        q_name = pose_dataset.get_basename(pose_dataset.q_frames_idxs[q_idx])# 'query_DJI_20231018092903_0016_D'        
        query_dir = os.path.join(render_dir, q_name)# 'data/temp/kings_college_refine/renderings/pt2_1_s0_sz320_theta2,0_t1,5_1,5_1,5/query_DJI_20231018092903_0016_D'

        for beam_i in range(n_beams):
            beam_dir = join(query_dir, f'beam_{beam_i}')

            rd = RenderedImagesDataset(beam_dir, verbose=False)        
            paths = rd.get_full_paths()
            candidates_pathlist += paths
            # (Pdb) candidates_pathlist[0]
            # 'data/temp/kings_college_refine/renderings/pt2_1_s0_sz320_theta2,0_t1,5_1,5_1,5/query_DJI_20231018092903_0016_D/beam_0/query_DJI_20231018092903_0016_D_0beam0.png'
    # return last query res; assumes they are all the same
    query_tensor = pose_dataset[pose_dataset.q_frames_idxs[q_idx]]['im']
    query_res = tuple(query_tensor.shape[-2:])
    
    return candidates_pathlist, query_res
