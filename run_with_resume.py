"""
训练中断后恢复训练的示例
"""
from recbole.quick_start import run_recbole
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_logger, get_model, get_trainer, init_seed

def resume_training():
    """从checkpoint恢复训练"""
    
    # 1. 配置参数（与原始训练保持一致）
    parameter_dict = {
        'train_neg_sample_args': None,
        'checkpoint_dir': 'saved',  # checkpoint保存目录
        'gpu_id': 0,  # 使用第7张GPU卡
        'use_gpu': True,  # 启用GPU
    }
    
    # 2. 初始化配置
    config = Config(
        model='SASRec',
        dataset='yambda',
        config_file_list=['configs/simple_dataset.yaml'],
        config_dict=parameter_dict
    )
    
    # 3. 初始化
    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    
    # 4. 数据准备
    dataset = create_dataset(config)
    train_data, valid_data, test_data = data_preparation(config, dataset)
    
    # 5. 模型初始化
    model = get_model(config['model'])(config, train_data._dataset).to(config['device'])
    
    # 6. Trainer初始化
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    # 7. 加载checkpoint（重点！）
    # 替换为你实际的checkpoint路径
    checkpoint_file = '/home/hongminjie/RecBole/saved/SASRec-Dec-09-2025_02-40-06.pth'  # 修改为实际路径
    trainer.resume_checkpoint(checkpoint_file)
    
    # 8. 继续训练
    best_valid_score, best_valid_result = trainer.fit(
        train_data, 
        valid_data, 
        saved=True, 
        show_progress=config['show_progress']
    )
    
    # 9. 测试
    test_result = trainer.evaluate(
        test_data, 
        load_best_model=True, 
        show_progress=config['show_progress']
    )
    
    print(f'Best valid result: {best_valid_result}')
    print(f'Test result: {test_result}')
    
    return best_valid_score, best_valid_result, test_result


if __name__ == '__main__':
    resume_training()
