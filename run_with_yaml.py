"""
使用YAML配置文件运行数据集
"""
from recbole.quick_start import run_recbole

if __name__ == '__main__':
    parameter_dict = {
        'train_neg_sample_args': None,
    }
    
    run_recbole(
        model='GRU4Rec', 
        dataset='yambda',  # 替换为你的数据集名称
        config_file_list=['configs/simple_dataset.yaml'],
        config_dict=parameter_dict
    )
