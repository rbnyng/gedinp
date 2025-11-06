"""
Test script for ConvCNP implementation.

Verifies model architecture, forward pass, and basic functionality.
"""

import torch
import numpy as np
from models.unet import UNet, UNetSmall
from models.convcnp import GEDIConvCNP, convcnp_loss, compute_metrics


def test_unet():
    """Test UNet architecture."""
    print("=" * 80)
    print("Testing UNet Architecture")
    print("=" * 80)

    # Test standard UNet
    model = UNet(in_channels=130, feature_dim=128, base_channels=32, depth=3)

    # Test input
    B, H, W = 2, 256, 256
    x = torch.randn(B, 130, H, W)

    # Forward pass
    out = model(x)

    print(f"✓ UNet forward pass successful")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {out.shape}")
    print(f"  Parameters:   {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Test small UNet
    model_small = UNetSmall(in_channels=130, feature_dim=128)
    out_small = model_small(x)

    print(f"✓ UNetSmall forward pass successful")
    print(f"  Output shape: {out_small.shape}")
    print(f"  Parameters:   {sum(p.numel() for p in model_small.parameters()) / 1e6:.2f}M")
    print()

    return True


def test_convcnp():
    """Test ConvCNP model."""
    print("=" * 80)
    print("Testing ConvCNP Model")
    print("=" * 80)

    # Initialize model
    model = GEDIConvCNP(
        embedding_channels=128,
        feature_dim=128,
        base_channels=32,
        unet_depth=3,
        decoder_hidden_dim=128,
        output_uncertainty=True,
        use_small_unet=False
    )

    print(f"✓ Model initialized")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Test input
    B, H, W = 2, 256, 256
    tile_embedding = torch.randn(B, 128, H, W)
    context_agbd = torch.randn(B, 1, H, W) * 0.5  # Normalized
    context_mask = torch.randint(0, 2, (B, 1, H, W)).float()

    # Forward pass
    pred_mean, pred_log_var = model(tile_embedding, context_agbd, context_mask)

    print(f"✓ Forward pass successful")
    print(f"  Input shapes:")
    print(f"    tile_embedding: {tile_embedding.shape}")
    print(f"    context_agbd:   {context_agbd.shape}")
    print(f"    context_mask:   {context_mask.shape}")
    print(f"  Output shapes:")
    print(f"    pred_mean:      {pred_mean.shape}")
    print(f"    pred_log_var:   {pred_log_var.shape}")

    # Test predict method
    pred_mean_out, pred_std_out = model.predict(tile_embedding, context_agbd, context_mask)

    print(f"✓ Predict method successful")
    print(f"  pred_mean: {pred_mean_out.shape}")
    print(f"  pred_std:  {pred_std_out.shape}")
    print()

    return True


def test_loss():
    """Test loss function."""
    print("=" * 80)
    print("Testing Loss Function")
    print("=" * 80)

    B, H, W = 2, 256, 256

    pred_mean = torch.randn(B, 1, H, W)
    pred_log_var = torch.randn(B, 1, H, W)
    target_agbd = torch.randn(B, 1, H, W)
    target_mask = torch.randint(0, 2, (B, 1, H, W)).float()

    # Test with uncertainty
    loss = convcnp_loss(pred_mean, pred_log_var, target_agbd, target_mask)

    print(f"✓ Loss computation successful")
    print(f"  Loss value: {loss.item():.4f}")

    # Test without uncertainty
    loss_no_var = convcnp_loss(pred_mean, None, target_agbd, target_mask)

    print(f"✓ Loss without uncertainty successful")
    print(f"  Loss value: {loss_no_var.item():.4f}")
    print()

    return True


def test_metrics():
    """Test metrics computation."""
    print("=" * 80)
    print("Testing Metrics")
    print("=" * 80)

    B, H, W = 2, 256, 256

    pred_mean = torch.randn(B, 1, H, W)
    pred_std = torch.abs(torch.randn(B, 1, H, W))
    target_agbd = torch.randn(B, 1, H, W)
    target_mask = torch.randint(0, 2, (B, 1, H, W)).float()

    # Compute metrics
    metrics = compute_metrics(pred_mean, pred_std, target_agbd, target_mask)

    print(f"✓ Metrics computation successful")
    print(f"  Metrics: {metrics}")
    print()

    return True


def test_gradient_flow():
    """Test gradient flow through model."""
    print("=" * 80)
    print("Testing Gradient Flow")
    print("=" * 80)

    model = GEDIConvCNP(
        embedding_channels=128,
        feature_dim=64,
        base_channels=16,
        unet_depth=2,
        use_small_unet=True
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # Small batch
    B, H, W = 1, 128, 128
    tile_embedding = torch.randn(B, 128, H, W)
    context_agbd = torch.randn(B, 1, H, W) * 0.5
    context_mask = torch.ones(B, 1, H, W)
    target_agbd = torch.randn(B, 1, H, W) * 0.5
    target_mask = torch.ones(B, 1, H, W)

    # Forward + backward
    optimizer.zero_grad()
    pred_mean, pred_log_var = model(tile_embedding, context_agbd, context_mask)
    loss = convcnp_loss(pred_mean, pred_log_var, target_agbd, target_mask)
    loss.backward()

    # Check gradients
    has_grads = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for _ in model.parameters())

    print(f"✓ Backward pass successful")
    print(f"  Loss: {loss.item():.4f}")
    print(f"  Gradients: {has_grads}/{total_params} parameters")

    # Optimizer step
    optimizer.step()
    print(f"✓ Optimizer step successful")
    print()

    return True


def test_variable_sizes():
    """Test model with different input sizes."""
    print("=" * 80)
    print("Testing Variable Input Sizes")
    print("=" * 80)

    model = GEDIConvCNP(
        embedding_channels=128,
        feature_dim=64,
        base_channels=16,
        unet_depth=2
    )

    sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]

    for H, W in sizes:
        tile_embedding = torch.randn(1, 128, H, W)
        context_agbd = torch.randn(1, 1, H, W)
        context_mask = torch.randint(0, 2, (1, 1, H, W)).float()

        pred_mean, pred_log_var = model(tile_embedding, context_agbd, context_mask)

        assert pred_mean.shape == (1, 1, H, W), f"Wrong output shape for {H}x{W}"
        print(f"✓ Size {H}x{W}: OK")

    print()
    return True


def main():
    """Run all tests."""
    print("\n")
    print("╔" + "=" * 78 + "╗")
    print("║" + " " * 20 + "ConvCNP Implementation Tests" + " " * 30 + "║")
    print("╚" + "=" * 78 + "╝")
    print()

    tests = [
        ("UNet Architecture", test_unet),
        ("ConvCNP Model", test_convcnp),
        ("Loss Function", test_loss),
        ("Metrics", test_metrics),
        ("Gradient Flow", test_gradient_flow),
        ("Variable Sizes", test_variable_sizes),
    ]

    passed = 0
    failed = 0

    for name, test_fn in tests:
        try:
            if test_fn():
                passed += 1
            else:
                failed += 1
                print(f"✗ {name} FAILED")
        except Exception as e:
            failed += 1
            print(f"✗ {name} FAILED with error: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    print(f"Passed: {passed}/{len(tests)}")
    print(f"Failed: {failed}/{len(tests)}")
    print("=" * 80)

    if failed == 0:
        print("\n🎉 All tests passed! ConvCNP implementation is ready.")
    else:
        print(f"\n⚠ {failed} test(s) failed. Please fix before using.")

    return failed == 0


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
