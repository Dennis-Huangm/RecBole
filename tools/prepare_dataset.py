"""
数据集格式转换工具
将你的数据转换为RecBole格式
"""
import pandas as pd
import os

def convert_to_recbole_format(
    input_file,
    output_dir,
    dataset_name,
    user_col='uid',
    item_col='item_id',
    time_col='timestamp',
    separator=',',
    extra_cols=None
):
    """
    将CSV/TSV/Parquet文件转换为RecBole格式
    
    Args:
        input_file: 输入文件路径 (CSV、TSV或Parquet)
        output_dir: 输出目录 (例如: 'dataset/your_dataset_name')
        dataset_name: 数据集名称
        user_col: 用户ID列名 (默认: 'uid')
        item_col: 物品ID列名 (默认: 'item_id')
        time_col: 时间戳列名 (默认: 'timestamp')
        separator: 输入文件分隔符 (',' 或 '\t')，仅对CSV/TSV有效
        extra_cols: 额外要保留的列名列表 (例如: ['is_organic', 'played_ratio_pct'])
    """
    # 读取数据
    print(f"正在读取文件: {input_file}")
    if input_file.endswith('.parquet'):
        df = pd.read_parquet(input_file)
        print(f"✓ 成功读取 Parquet 文件")
    else:
        df = pd.read_csv(input_file, sep=separator)
        print(f"✓ 成功读取 CSV/TSV 文件")
    
    # 检查必需列是否存在
    required_cols = [user_col, item_col, time_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"缺少必需列: {missing_cols}")
    
    # 选择需要的列
    cols_to_keep = [user_col, item_col, time_col]
    if extra_cols:
        for col in extra_cols:
            if col in df.columns:
                cols_to_keep.append(col)
            else:
                print(f"⚠ 警告: 额外列 '{col}' 不存在于数据中")
    
    df = df[cols_to_keep]
    
    # 重命名列（如果需要）
    rename_dict = {user_col: 'user_id', item_col: 'item_id', time_col: 'timestamp'}
    df = df.rename(columns=rename_dict)
    
    # 数据统计
    print("\n数据集统计:")
    print(f"  总交互数: {len(df)}")
    print(f"  用户数: {df['user_id'].nunique()}")
    print(f"  物品数: {df['item_id'].nunique()}")
    print(f"  稀疏度: {1 - len(df) / (df['user_id'].nunique() * df['item_id'].nunique()):.4f}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存为RecBole格式
    output_file = os.path.join(output_dir, f'{dataset_name}.inter')
    print(f"\n正在保存到: {output_file}")
    
    # 构建表头
    header_parts = ['user_id:token', 'item_id:token', 'timestamp:float']
    if extra_cols:
        for col in extra_cols:
            if col in df.columns:
                # 根据数据类型推断字段类型
                if df[col].dtype in ['float64', 'float32']:
                    header_parts.append(f'{col}:float')
                elif df[col].dtype in ['int64', 'int32', 'uint32', 'uint16', 'uint8']:
                    header_parts.append(f'{col}:token')
                else:
                    header_parts.append(f'{col}:token')
    
    # 使用批量写入提升性能
    with open(output_file, 'w', encoding='utf-8') as f:
        # 写入表头
        f.write('\t'.join(header_parts) + '\n')
        
        # 批量写入数据（比 iterrows 快很多）
        print(f"正在写入 {len(df)} 行数据...")
        batch_size = 100000
        for start_idx in range(0, len(df), batch_size):
            end_idx = min(start_idx + batch_size, len(df))
            batch = df.iloc[start_idx:end_idx]
            
            # 转换为字符串并写入
            lines = []
            for _, row in batch.iterrows():
                line_parts = [str(row['user_id']), str(row['item_id']), str(row['timestamp'])]
                if extra_cols:
                    for col in extra_cols:
                        if col in df.columns:
                            line_parts.append(str(row[col]))
                lines.append('\t'.join(line_parts))
            
            f.write('\n'.join(lines) + '\n')
            
            if (end_idx % 1000000) == 0 or end_idx == len(df):
                print(f"  已写入 {end_idx}/{len(df)} 行 ({end_idx/len(df)*100:.1f}%)")
    
    print(f"✓ 转换完成! 文件保存在: {output_file}")
    
    # 打印示例数据
    print("\n前5行数据:")
    with open(output_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < 6:  # 表头 + 5行数据
                print(f"  {line.strip()}")
            else:
                break


def validate_recbole_format(inter_file):
    """
    验证.inter文件格式是否正确
    
    Args:
        inter_file: .inter文件路径
    """
    print(f"正在验证文件: {inter_file}")
    
    with open(inter_file, 'r', encoding='utf-8') as f:
        # 读取表头
        header = f.readline().strip()
        print(f"表头: {header}")
        
        # 检查表头格式
        expected_header = 'user_id:token\titem_id:token\ttimestamp:float'
        if header != expected_header:
            print(f"⚠ 警告: 表头格式可能不正确")
            print(f"  期望: {expected_header}")
            print(f"  实际: {header}")
        else:
            print("✓ 表头格式正确")
        
        # 读取前几行数据
        print("\n前5行数据:")
        for i in range(5):
            line = f.readline().strip()
            if not line:
                break
            parts = line.split('\t')
            if len(parts) != 3:
                print(f"⚠ 警告: 第{i+2}行字段数不正确: {parts}")
            else:
                print(f"  行{i+2}: user={parts[0]}, item={parts[1]}, time={parts[2]}")
    
    print("\n✓ 验证完成")


# 使用示例
if __name__ == '__main__':
    # 示例1: 从Parquet转换（当前数据集）
    # convert_to_recbole_format(
    #     input_file='listens.parquet',
    #     output_dir='dataset/listens',
    #     dataset_name='listens',
    #     user_col='uid',
    #     item_col='item_id',
    #     time_col='timestamp',
    #     extra_cols=['is_organic', 'played_ratio_pct', 'track_length_seconds']  # 可选：保留额外列
    # )
    
    # 示例2: 只保留必需列
    # convert_to_recbole_format(
    #     input_file='listens.parquet',
    #     output_dir='dataset/listens',
    #     dataset_name='listens',
    #     user_col='uid',
    #     item_col='item_id',
    #     time_col='timestamp',
    #     extra_cols=None  # 不保留额外列
    # )
    
    # 示例3: 验证格式
    validate_recbole_format('dataset/listens/listens.inter')
