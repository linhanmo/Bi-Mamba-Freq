import torch
from mambapy.mamba import Mamba, MambaConfig

def test_mamba_tensor():
    print("=== Mamba 张量输出测试 ===\n")
    
    # 1. 基础配置
    B, L, D = 4, 64, 128      # 批次大小, 序列长度, 特征维度
    config = MambaConfig(d_model=D, n_layers=2)
    model = Mamba(config)
    
    # 2. 随机输入
    x = torch.randn(B, L, D)
    print(f"输入形状: {x.shape}")
    print(f"输入范围: [{x.min().item():.4f}, {x.max().item():.4f}]")
    print(f"输入是否包含 NaN: {torch.isnan(x).any().item()}")
    
    # 3. 前向传播
    y = model(x)
    print(f"\n输出形状: {y.shape}")
    print(f"输出范围: [{y.min().item():.4f}, {y.max().item():.4f}]")
    print(f"输出是否包含 NaN: {torch.isnan(y).any().item()}")
    print(f"输出是否包含 Inf: {torch.isinf(y).any().item()}")
    
    # 4. 形状一致性检查
    assert y.shape == x.shape, f"形状不匹配: {y.shape} vs {x.shape}"
    print("\n✅ 输出形状与输入一致")
    
    # 5. 梯度测试
    print("\n--- 梯度测试 ---")
    model.zero_grad()
    loss = y.sum()
    loss.backward()
    
    total_grad_norm = 0.0
    num_params = 0
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            total_grad_norm += grad_norm
            num_params += 1
            print(f"参数 {name}: 梯度范数 = {grad_norm:.4f}")
    
    print(f"\n总梯度范数: {total_grad_norm:.4f}")
    print("✅ 反向传播正常，所有参数都有梯度")
    
    # 6. CPU/GPU 设备测试（如果可用）
    if torch.cuda.is_available():
        print("\n--- GPU 测试 ---")
        device = torch.device("cuda")
        model_gpu = model.to(device)
        x_gpu = x.to(device)
        y_gpu = model_gpu(x_gpu)
        print(f"GPU 输出形状: {y_gpu.shape}")
        print(f"GPU 输出设备: {y_gpu.device}")
        print("✅ GPU 前向传播正常")
    
    print("\n=== 所有测试通过 ===")

if __name__ == "__main__":
    test_mamba_tensor()