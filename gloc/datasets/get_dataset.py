from os.path import join
import torchvision.transforms as T

from gloc.datasets import PoseDataset
from gloc.datasets.dataset_nolabels import IntrinsicsDataset


def get_dataset(name, paths_conf, transform=None):
    # if 'Aachen' in name:
    if name in ['Aachen_night', 'Aachen_day', 'Aachen_real', 'Aachen_real_und']:
        dataset = IntrinsicsDataset(name, paths_conf, transform)
    else:
        # 这里paths_conf是paths_conf[DS]
        dataset = PoseDataset(name, paths_conf, transform)
    
    return dataset


def get_transform(args, colmap_dir=''):
    res = args.res
    
    if args.feat_model == 'Dinov2' or args.feat_model == 'Dinov2_contrast' or args.feat_model == 'DepthV2':
        # colmap_dir是data/all_colmaps/UAVD4L-LoD/inTraj/colmap_320
        # 拼接得到相机参数文件路径
        cam_file = join(colmap_dir, 'cameras.txt')
        # 读取 cameras.txt 文件第11行（下标10），并按空格分割，得到相机参数
        # 第11行是相机参数，格式为：相机ID 相机模型 宽 高 焦距x 焦距y 主点x 主点y
        random_line = open(cam_file, 'r').readlines()[10].split(' ')
        w, h = int(random_line[2]), int(random_line[3])
        # 无缩放
        scale_factor = 1

        patch_size = 14
        new_h = patch_size * ((h * scale_factor) // patch_size)
        new_w = patch_size * ((w * scale_factor) // patch_size)

        new_h =  480 #int(new_h)  480 Swiss 448  Video 480 Japan_02 1080 Japan_02_one_third 360
        new_w =  720 #int(new_w) 改 720 Swiss  602  Video 270 Japan_02 1920 Japan_02_one_third 640

        # import torchvision.transforms as T
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((new_h, new_w), antialias=True),
            # T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])        

    elif ('Aachen' not in args.name) and (colmap_dir != ''):            
        cam_file = join(colmap_dir, 'cameras.txt')                                                                                                
        random_line = open(cam_file, 'r').readlines()[10].split(' ')
        w, h = int(random_line[2]), int(random_line[3])
        ratio = min(h, w) / res
        new_h = int(h/ratio)
        new_w = int(w/ratio)
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((new_h, new_w), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Resize(res, antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform
