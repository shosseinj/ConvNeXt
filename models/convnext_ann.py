def compute_ann_macs_manual(input_size=32):
    """
    Manually compute MACs for ConvNeXt ANN.
    Architecture: [96, 192, 384, 768], blocks: [3, 3, 9, 3]
    
    Args:
        input_size: 32 for CIFAR-10, 224 for ImageNet
    """
    total_macs = 0
    
    # Calculate spatial sizes based on input
    # Stem: stride=4
    H0 = W0 = input_size // 4
    # Downsample 1: stride=2
    H1 = W1 = H0 // 2
    # Downsample 2: stride=2
    H2 = W2 = H1 // 2
    # Downsample 3: stride=2
    H3 = W3 = H2 // 2
    
    print(f"Input: {input_size}×{input_size}")
    print(f"Spatial sizes: {H0}×{H0} -> {H1}×{H1} -> {H2}×{H2} -> {H3}×{H3}")
    print()
    
    # ================================================================
    # STEM: Conv2d(3, 96, kernel=4, stride=4)
    # ================================================================
    stem_macs = 4 * 4 * 3 * 96 * H0 * W0
    total_macs += stem_macs
    print(f"Stem:                {stem_macs/1e6:>10.4f} M MACs")
    
    # ================================================================
    # STAGE 0: 96 channels, H0×W0, 3 blocks
    # ================================================================
    C, H, W = 96, H0, W0
    for b in range(3):
        dw_macs = 7 * 7 * C * H * W
        pw1_macs = 1 * 1 * C * (4*C) * H * W
        pw2_macs = 1 * 1 * (4*C) * C * H * W
        block_macs = dw_macs + pw1_macs + pw2_macs
        total_macs += block_macs
        print(f"Stage 0, Block {b}:    {block_macs/1e6:>10.4f} M MACs")
    
    # ================================================================
    # DOWNSAMPLE 1: Conv2d(96, 192, kernel=2, stride=2)
    # ================================================================
    ds1_macs = 2 * 2 * 96 * 192 * H1 * W1
    total_macs += ds1_macs
    print(f"Downsample 1:        {ds1_macs/1e6:>10.4f} M MACs")
    
    # ================================================================
    # STAGE 1: 192 channels, H1×W1, 3 blocks
    # ================================================================
    C, H, W = 192, H1, W1
    for b in range(3):
        dw_macs = 7 * 7 * C * H * W
        pw1_macs = 1 * 1 * C * (4*C) * H * W
        pw2_macs = 1 * 1 * (4*C) * C * H * W
        block_macs = dw_macs + pw1_macs + pw2_macs
        total_macs += block_macs
        print(f"Stage 1, Block {b}:    {block_macs/1e6:>10.4f} M MACs")
    
    # ================================================================
    # DOWNSAMPLE 2: Conv2d(192, 384, kernel=2, stride=2)
    # ================================================================
    ds2_macs = 2 * 2 * 192 * 384 * H2 * W2
    total_macs += ds2_macs
    print(f"Downsample 2:        {ds2_macs/1e6:>10.4f} M MACs")
    
    # ================================================================
    # STAGE 2: 384 channels, H2×W2, 9 blocks
    # ================================================================
    C, H, W = 384, H2, W2
    for b in range(9):
        dw_macs = 7 * 7 * C * H * W
        pw1_macs = 1 * 1 * C * (4*C) * H * W
        pw2_macs = 1 * 1 * (4*C) * C * H * W
        block_macs = dw_macs + pw1_macs + pw2_macs
        total_macs += block_macs
        print(f"Stage 2, Block {b}:    {block_macs/1e6:>10.4f} M MACs")
    
    # ================================================================
    # DOWNSAMPLE 3: Conv2d(384, 768, kernel=2, stride=2)
    # ================================================================
    ds3_macs = 2 * 2 * 384 * 768 * H3 * W3
    total_macs += ds3_macs
    print(f"Downsample 3:        {ds3_macs/1e6:>10.4f} M MACs")
    
    # ================================================================
    # STAGE 3: 768 channels, H3×W3, 3 blocks
    # ================================================================
    C, H, W = 768, H3, W3
    for b in range(3):
        dw_macs = 7 * 7 * C * H * W
        pw1_macs = 1 * 1 * C * (4*C) * H * W
        pw2_macs = 1 * 1 * (4*C) * C * H * W
        block_macs = dw_macs + pw1_macs + pw2_macs
        total_macs += block_macs
        print(f"Stage 3, Block {b}:    {block_macs/1e6:>10.4f} M MACs")
    
    # ================================================================
    # HEAD: Linear(768 -> 10)
    # ================================================================
    head_macs = 768 * 10
    total_macs += head_macs
    print(f"Head:                {head_macs/1e6:>10.4f} M MACs")
    
    # ================================================================
    # TOTAL
    # ================================================================
    print(f"\n{'='*60}")
    print(f"Input size: {input_size}×{input_size}")
    print(f"Total ANN MACs: {total_macs/1e6:.2f} M = {total_macs/1e9:.4f} G")
    print(f"Total ANN Energy: {total_macs * 4.6e-12 * 1000:.4f} mJ")
    print(f"{'='*60}")
    
    return total_macs


# ================================================================
# Run for BOTH input sizes
# ================================================================
if __name__ == '__main__':
    print("="*60)
    print("ANN MACs for CIFAR-10 (32×32):")
    print("="*60)
    macs_32 = compute_ann_macs_manual(input_size=32)
    
    print("\n")
    print("="*60)
    print("ANN MACs for ImageNet (224×224):")
    print("="*60)
    macs_224 = compute_ann_macs_manual(input_size=224)