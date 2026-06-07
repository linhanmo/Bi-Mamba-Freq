"""
CSI300 日频数据预处理脚本
输入: CSI300_2008_2025.csv (包含 date,Open,High,Low,Close,Volume)
输出: csi300_processed.npz (包含 X_train, y_train_cls, y_train_reg, ...)
符合 Bi-Mamba-Freq 实验设计: L=60, H=5, 动态阈值三分类 + 波动率回归
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse
import warnings
warnings.filterwarnings('ignore')

# ==================== 配置参数 ====================
SEQ_LEN = 60          # 回看窗口长度（交易日）
HORIZON = 5           # 预测步长（未来第5天）
VOL_THRESHOLD = 1.5   # 动态阈值倍数（基于滚动20日波动率）
VOL_WINDOW = 20       # 波动率滚动窗口

# ==================== 1. 数据加载与基础清洗 ====================
def load_and_clean(file_path):
    """加载CSV，处理日期索引和缺失值"""
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    df.sort_values('date', inplace=True)
    df.set_index('date', inplace=True)
    
    # 只保留需要的列
    df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
    # 列名统一为小写（便于后续操作）
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    
    # 前向填充缺失值，再删除仍有缺失的行（如第一天）
    df.ffill(inplace=True)
    df.dropna(inplace=True)
    
    # 删除重复索引
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"[CSI300] 原始数据行数: {len(df)}")
    return df

# ==================== 2. 特征工程 ====================
def add_features(df):
    """添加收益率、波动率、RSI、成交量比率等特征"""
    # 收益率
    df['return'] = df['close'].pct_change()
    df['log_return'] = np.log(df['close'] / df['close'].shift(1))
    
    # 滚动波动率（历史波动率，用于动态阈值）
    df['rolling_vol'] = df['return'].rolling(VOL_WINDOW).std()
    
    # 技术指标：RSI(14)
    delta = df['close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta).clip(lower=0).rolling(14).mean()
    rs = gain / loss
    df['rsi_14'] = 100 - (100 / (1 + rs))
    
    # 成交量比率（相对于5日均量）
    df['volume_ma5'] = df['volume'].rolling(5).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma5']
    
    # 价格位置特征（可选）
    df['high_low_ratio'] = (df['high'] - df['low']) / df['close']
    
    # 删除因计算产生的NaN
    df.dropna(inplace=True)
    print(f"[CSI300] 特征工程后行数: {len(df)}")
    return df

# ==================== 3. 标签生成（主任务+辅助任务） ====================
def add_labels(df):
    """生成主任务三分类标签和辅助任务未来波动率标签"""
    # 未来 HORIZON 天的收益率
    df['future_return'] = df['close'].shift(-HORIZON) / df['close'] - 1
    
    # 动态阈值 = 滚动波动率 * 倍数
    df['threshold'] = df['rolling_vol'] * VOL_THRESHOLD
    
    # 主任务：三分类标签 (-1, 0, 1)
    df['label_cls'] = 0
    df.loc[df['future_return'] > df['threshold'], 'label_cls'] = 1
    df.loc[df['future_return'] < -df['threshold'], 'label_cls'] = -1
    
    # 辅助任务：未来 HORIZON 天的已实现波动率
    future_returns = []
    for i in range(1, HORIZON+1):
        future_returns.append(df['log_return'].shift(-i))
    future_returns_df = pd.concat(future_returns, axis=1)
    df['future_volatility'] = np.sqrt(np.mean(np.square(future_returns_df), axis=1))
    
    # 删除无法计算标签的行（最后 HORIZON 天）
    df.dropna(inplace=True)
    print(f"[CSI300] 标签生成后行数: {len(df)}")
    return df

# ==================== 4. 划分训练/验证/测试集（时间顺序） ====================
def split_time_series(df, train_ratio=0.7, val_ratio=0.1):
    """按时间顺序划分，训练:验证:测试 = 70:10:20"""
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    train_df = df.iloc[:train_end]
    val_df = df.iloc[train_end:val_end]
    test_df = df.iloc[val_end:]
    print(f"[CSI300] 训练集: {len(train_df)}, 验证集: {len(val_df)}, 测试集: {len(test_df)}")
    return train_df, val_df, test_df

# ==================== 5. 特征标准化 ====================
def normalize_features(train_df, val_df, test_df, feature_cols):
    """在训练集上拟合标准化参数，转换所有集合"""
    train_values = train_df[feature_cols].astype(np.float32)
    mean = train_values.mean(axis=0)
    std = train_values.std(axis=0, ddof=0)
    std = std.replace(0.0, 1.0)

    train_df_norm = train_df.copy()
    val_df_norm = val_df.copy()
    test_df_norm = test_df.copy()
    train_df_norm[feature_cols] = (train_df_norm[feature_cols].astype(np.float32) - mean) / std
    val_df_norm[feature_cols] = (val_df_norm[feature_cols].astype(np.float32) - mean) / std
    test_df_norm[feature_cols] = (test_df_norm[feature_cols].astype(np.float32) - mean) / std
    scaler = {"mean": mean.to_numpy(dtype=np.float32), "std": std.to_numpy(dtype=np.float32)}
    return train_df_norm, val_df_norm, test_df_norm, scaler

# ==================== 6. 滑动窗口生成序列样本 ====================
def create_sequences(df, feature_cols, label_col='label_cls', reg_label_col='future_volatility'):
    """生成输入序列 X 和对应标签 y_cls, y_reg"""
    X, y_cls, y_reg = [], [], []
    data = df[feature_cols].values
    labels_cls = df[label_col].values
    labels_reg = df[reg_label_col].values
    
    for i in range(len(df) - SEQ_LEN + 1):
        X.append(data[i:i+SEQ_LEN])
        # 标签对应窗口最后一个时间步的未来方向（已对齐）
        y_cls.append(labels_cls[i+SEQ_LEN-1])
        y_reg.append(labels_reg[i+SEQ_LEN-1])
    
    X = np.array(X, dtype=np.float32)
    y_cls = np.array(y_cls, dtype=np.int8)
    y_reg = np.array(y_reg, dtype=np.float32)
    print(f"[CSI300] 生成序列样本数: {len(X)}")
    return X, y_cls, y_reg

# ==================== 主流程 ====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[2]
    input_path = Path(args.input) if args.input else (project_root / "datasets" / "CSI300_2008_2025.csv")
    output_dir = Path(args.output_dir) if args.output_dir else (project_root / "data")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载与清洗
    df = load_and_clean(str(input_path))
    
    # 特征工程
    df = add_features(df)
    
    # 标签生成
    df = add_labels(df)
    
    # 定义特征列（不包括原始价格、辅助列、标签）
    exclude_cols = ['open', 'high', 'low', 'close', 'volume',
                    'return', 'log_return', 'rolling_vol', 'volume_ma5',
                    'future_return', 'threshold', 'label_cls', 'future_volatility']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    print(f"[CSI300] 使用特征列: {feature_cols}")
    
    # 时间顺序划分
    train_df, val_df, test_df = split_time_series(df)
    
    # 标准化
    train_df_norm, val_df_norm, test_df_norm, scaler = normalize_features(
        train_df, val_df, test_df, feature_cols
    )
    
    # 生成序列样本
    X_train, y_train_cls, y_train_reg = create_sequences(train_df_norm, feature_cols)
    X_val, y_val_cls, y_val_reg = create_sequences(val_df_norm, feature_cols)
    X_test, y_test_cls, y_test_reg = create_sequences(test_df_norm, feature_cols)
    
    # 保存为 npz 文件
    output_path = output_dir / "csi300_processed.npz"
    np.savez(str(output_path),
             X_train=X_train, y_train_cls=y_train_cls, y_train_reg=y_train_reg,
             X_val=X_val, y_val_cls=y_val_cls, y_val_reg=y_val_reg,
             X_test=X_test, y_test_cls=y_test_cls, y_test_reg=y_test_reg,
             feature_cols=feature_cols)
    print(f"[CSI300] 预处理完成，数据已保存至 {output_path}")
    
    # 打印标签分布统计
    print("\n=== CSI300 标签分布 ===")
    for name, y in [("训练集", y_train_cls), ("验证集", y_val_cls), ("测试集", y_test_cls)]:
        unique, counts = np.unique(y, return_counts=True)
        print(f"{name}: {dict(zip(unique, counts))}")

if __name__ == "__main__":
    main()
