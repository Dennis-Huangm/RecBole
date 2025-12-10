"""
加载已保存模型的示例
适用于：
1. 加载已完成训练的模型进行评估
2. 加载模型进行增量训练
"""
from recbole.quick_start import load_data_and_model
from recbole.utils import get_trainer

def load_and_evaluate():
    """加载模型并评估"""
    
    # 加载模型和数据（一步完成）
    # 替换为你实际的checkpoint路径
    checkpoint_file = 'saved/HGN-Dec-09-2025_11-00-00.pth'
    
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=checkpoint_file
    )
    
    # 创建trainer
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    # 评估模型
    test_result = trainer.evaluate(
        test_data, 
        load_best_model=False,  # 已经加载了模型，设为False
        show_progress=config['show_progress']
    )
    
    print(f'Test result: {test_result}')
    
    return test_result


def load_and_continue_training():
    """加载模型并继续训练"""
    
    checkpoint_file = 'saved/HGN-Dec-09-2025_11-00-00.pth'
    
    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
        model_file=checkpoint_file
    )
    
    # 创建trainer
    trainer = get_trainer(config['MODEL_TYPE'], config['model'])(config, model)
    
    # 注意：load_data_and_model不会恢复训练状态（epoch、optimizer等）
    # 如果需要完整恢复训练状态，请使用方法1中的resume_checkpoint
    
    # 继续训练
    best_valid_score, best_valid_result = trainer.fit(
        train_data, 
        valid_data, 
        saved=True, 
        show_progress=config['show_progress']
    )
    
    test_result = trainer.evaluate(
        test_data, 
        load_best_model=True, 
        show_progress=config['show_progress']
    )
    
    print(f'Best valid result: {best_valid_result}')
    print(f'Test result: {test_result}')
    
    return best_valid_score, best_valid_result, test_result


if __name__ == '__main__':
    # 选择一个运行
    load_and_evaluate()
    # load_and_continue_training()
