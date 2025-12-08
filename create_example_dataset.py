"""
创建一个示例数据集用于测试
"""
import os
import random
import time

def create_example_dataset(dataset_name='yambda'):
    """创建一个简单的示例数据集"""
    
    # 创建目录
    dataset_dir = f'dataset/{dataset_name}'
    os.makedirs(dataset_dir, exist_ok=True)
    
    # 参数设置
    num_users = 100
    num_items = 200
    num_interactions = 5000
    
    # 生成交互数据
    interactions = []
    base_time = int(time.time()) - 365 * 24 * 3600  # 一年前的时间戳
    
    for _ in range(num_interactions):
        user_id = random.randint(1, num_users)
        item_id = random.randint(1, num_items)
        timestamp = base_time + random.randint(0, 365 * 24 * 3600)
        interactions.append((user_id, item_id, timestamp))
    
    # 按时间排序
    interactions.sort(key=lambda x: x[2])
    
    # 保存到文件
    output_file = os.path.join(dataset_dir, f'{dataset_name}.inter')
    print(f"正在创建数据集: {output_file}")
    
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write('user_id:token\titem_id:token\ttimestamp:float\n')
        
        # 写入数据
        for user_id, item_id, timestamp in interactions:
            f.write(f'{user_id}\t{item_id}\t{float(timestamp)}\n')
    
    print(f"✓ 数据集创建完成!")
    print(f"  用户数: {num_users}")
    print(f"  物品数: {num_items}")
    print(f"  交互数: {num_interactions}")
    print(f"  文件位置: {output_file}")
    
    # 显示前几行
    print("\n前5行数据:")
    with open(output_file, 'r') as f:
        for i, line in enumerate(f):
            if i < 6:
                print(f"  {line.strip()}")
    
    return dataset_name

if __name__ == '__main__':
    dataset_name = create_example_dataset('example_dataset')
    
    print("\n接下来可以运行:")
    print(f"  python run_simple_dataset.py")
    print(f"  (需要先修改脚本中的 dataset='example_dataset')")
