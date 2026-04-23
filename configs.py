conf = {
    'cambridge': [
        {
            # Set N.1
            'beams': 2,
            'steps': 40,
            'N': 52,
            'M': 2,
            'protocol': '2_1',
            'center_std': [1.5,1.5, 1.5],
            'teta': [2],
            'gamma': 0.3,
            'res': 320,
            'colmap_res': 320,
        },      
    ]
}


def get_config(ds_name):
    cambridge_scenes = [
        'StMarysChurch', 'OldHospital', 'KingsCollege', 'ShopFacade','inTraj', 'outTraj', 
        'Synthesis', 'Swiss_in', 'Swiss_out', 'Video', 'Japan_02', 'Japan_02_one_third', 'Japan_06', 'Japan_07',
        'Seq1', 'Seq2', 'Seq3', 'Seq2_weather', 'Japan_06_new', 'Japan_07_new', 'Seq4', 'Seq5'
    ]
    
    if ds_name in cambridge_scenes:
        return conf['cambridge']
    else:
        return NotImplementedError
