import argparse
from os.path import join
from datetime import datetime
import os 


PROJECT_HOME = '/home/ubuntu/code/city-dreamer'

def parse_args():
    parser = argparse.ArgumentParser(description='Argument parser')
    # exp args
    parser.add_argument('--name', type=str, help='DS name', 
                        choices=['Aachen', 'Aachen_real', 'Aachen_day', 'Aachen_night', 'Aachen_real_und', 'Aachen_small', 
                                'KingsCollege', 'KingsCollege_und', 'StMarysChurch_und',
                                'ShopFacade', 'ShopFacade_und', 
                                'OldHospital', 'OldHospital_und',
                                'chess', 'office', 'fire', 'stairs', 'redkitchen', 'pumpkin', 'heads', 
                                'inTraj', 'outTraj', 'Synthesis', 'Swiss_in', 'Swiss_out', 
                                'Video', 'Japan_02', 'Japan_02_one_third', 'Japan_06', 'Japan_07',
                                'Seq1', 'Seq2', 'Seq3', 'Seq2_weather', 
                                'Japan_06_new', 'Japan_07_new', 'Seq4', 'Seq5'], default='inTraj')
    parser.add_argument('--exp_name', type=str, help='log folder', default='kings_college_refine')
    parser.add_argument('--res', type=int, help='resolution', default=320)
    parser.add_argument('--seed', type=int, help='seed', default=0)
    parser.add_argument('--first_step', type=int, help='start from', default=None)
    parser.add_argument('--hard_stop', type=int, help='interrupt at step N, but dont consider it for scaling noise', default=-1)
    parser.add_argument('--resume_step', type=str, help='resume folder', default=None)
    parser.add_argument('--save_feats', action='store_true', help='seed', default=False)
    parser.add_argument('--pose_prior', type=str, help='start from a pose prior in this file', default='/home/czy98/psb/LIB/mcloc_pure_code/mcloc_pure_code/data/UAVD4L-LoD/inTraj/GPS_pose_new_all.txt')
    parser.add_argument('--pt_base_path', type=str, help='base path for .pt files containing instance segmentation data', default='/home/amax/LoD-Loc v3_temp/mcloc_pure_code/Ins_data/Japan_07/PT_640_360_09091800/conf_0.3')
    parser.add_argument('--clean_logs', action='store_true', help='remove renderings in the end', default=False)
    parser.add_argument('--chunk_size', type=int, help='n feats at a time', default=1100)

    # path args
    parser.add_argument("--storage_dir", type=str, default='/storage/gtrivigno/vloc/renderings', help='model path')
    parser.add_argument("--fix_storage", action='store_true', default=False, help='model path')

    # render args
    parser.add_argument('--colmap_res', type=int, help='res', default=320)    
    parser.add_argument('--mesh', type=str, help='mesh type', choices=['colored', 'colored_14', 'colored_15', 'textured'], default='colored')    
    parser.add_argument('--renderer', type=str, help='renderer type', choices=['o3d', 'nerf', 'g_splatting'], default='o3d')    

    # perturb args
    parser.add_argument('-pt', '--protocol', type=str, help='protocol', 
                        choices=['1_0', '1_1', '2_0', '2_1'], default='2_1')
    parser.add_argument('--sampler', type=str, help='sampler', default='rand_yaw_or_pitch',
                        choices=['rand', 'rand_yaw_or_pitch', 'rand_yaw_and_pitch', 'rand_and_yaw_and_pitch'])    
    parser.add_argument('--beams', type=int, help='N. beams to optimize independetly', default=1)
    parser.add_argument('--steps', type=int, help='iterations', default=20)
    parser.add_argument('--teta', nargs='+', type=float, help='max angle', default=[1])
    parser.add_argument('--center_std', nargs='+', type=float, default=[1., 1., 1.])
    parser.add_argument('--N', type=int, help='N views to render in total, per query', default=20)

    parser.add_argument('--M', type=int, help='In each beam, perturb the first M rather than only first cand.', default=4)
    parser.add_argument('--gamma', type=float, help='min scale', default=0.1)

    #-exp_name kings_college_refine --renderer g_splatting --clean_logs
    
    # eval scripts
    parser.add_argument("--eval_renders_dir", type=str, default='', help='eval render dir')

    # random args
    parser.add_argument('--only_step', type=int, help='iterations', default=-1)

    # Render Config
    parser.add_argument("--render_config", type=str, default='./config/config_RealTime_render_1_in.json', help='Render Config')

    #OSM
    parser.add_argument("--osm_dir", default=os.path.join(PROJECT_HOME, "data", "xml"))
    parser.add_argument("--ges_dir", default=os.path.join(PROJECT_HOME, "data", "ges"))
    parser.add_argument(
        "--seg_dir", default=os.path.join(PROJECT_HOME, "data", "ges", "%s", "seg")
    )
    parser.add_argument(
        "--osm_out_dir", default=os.path.join(PROJECT_HOME, "data", "osm", 'US-NewYork')
    )
    parser.add_argument(
        "--ges_out_dir",
        default=os.path.join(PROJECT_HOME, "data", "ges", "%s", "raycasting"),
    )
    parser.add_argument("--patch_size", default=1536)
    parser.add_argument("--max_height", default=640)
    parser.add_argument("--zoom", default=18)
    parser.add_argument("--debug", action="store_true")

    args = parser.parse_args()
    args.save_dir = join("logs", args.exp_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    
    ## some consistency checks
    assert args.N % args.beams == 0, 'N (total views to rend) has to be a multiple of N. beams'
    if args.mesh == 'textured' and not args.name.startswith('Aachen'):
        raise ValueError('Textured mesh is only available for Aachen')
    if args.protocol[0] not in ['0', '1']:
        assert args.N % args.M == 0, f'In protocol 2, N ({args.N}) has to be a multiple of M ({args.M})'
        assert (args.N // args.beams) % args.M == 0, f'In protocols with M!=1, N/beams ({args.N//args.beams}) has to be a multiple of M ({args.M})'
    if 'yaw_and_pitch' in args.sampler:
        assert len(args.teta) == 2, f'Sampler {args.sampler} requires 2 angles, 1 for yaw, 1 for pitch'
    else:
        assert len(args.teta) == 1, f'Sampler {args.sampler} requires only 1 angle'        

    return args
