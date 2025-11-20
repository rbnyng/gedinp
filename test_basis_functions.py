"""
Test script for basis function implementation.

This script verifies that the BasisFunctionEncoder and GEDINeuralProcess
work correctly with different basis function types.
"""

import torch
from models.neural_process import BasisFunctionEncoder, GEDINeuralProcess


def test_basis_function_encoder():
    """Test BasisFunctionEncoder with different types."""
    print("Testing BasisFunctionEncoder...")

    batch_size = 10
    coords = torch.randn(batch_size, 2)

    # Test 'none' (baseline)
    print("\n1. Testing 'none' (raw coordinates)...")
    encoder_none = BasisFunctionEncoder(coord_dim=2, basis_type='none')
    output_none = encoder_none(coords)
    assert output_none.shape == (batch_size, 2), f"Expected shape (10, 2), got {output_none.shape}"
    assert torch.allclose(output_none, coords), "Raw coords should be unchanged"
    print(f"   ✓ Output shape: {output_none.shape}")

    # Test 'fourier_random'
    print("\n2. Testing 'fourier_random'...")
    encoder_fourier = BasisFunctionEncoder(
        coord_dim=2,
        basis_type='fourier_random',
        num_frequencies=32
    )
    output_fourier = encoder_fourier(coords)
    expected_dim = 2 * 32  # sin and cos for each frequency
    assert output_fourier.shape == (batch_size, expected_dim), \
        f"Expected shape (10, {expected_dim}), got {output_fourier.shape}"
    print(f"   ✓ Output shape: {output_fourier.shape}")
    print(f"   ✓ Frequency matrix shape: {encoder_fourier.frequency_matrix.shape}")

    # Test 'fourier_learnable'
    print("\n3. Testing 'fourier_learnable'...")
    encoder_learnable = BasisFunctionEncoder(
        coord_dim=2,
        basis_type='fourier_learnable',
        num_frequencies=16
    )
    output_learnable = encoder_learnable(coords)
    expected_dim = 2 * 16
    assert output_learnable.shape == (batch_size, expected_dim), \
        f"Expected shape (10, {expected_dim}), got {output_learnable.shape}"
    assert isinstance(encoder_learnable.frequency_matrix, torch.nn.Parameter), \
        "Frequency matrix should be a Parameter"
    print(f"   ✓ Output shape: {output_learnable.shape}")
    print(f"   ✓ Frequency matrix is learnable: {encoder_learnable.frequency_matrix.requires_grad}")

    # Test 'hybrid'
    print("\n4. Testing 'hybrid' (raw + Fourier)...")
    encoder_hybrid = BasisFunctionEncoder(
        coord_dim=2,
        basis_type='hybrid',
        num_frequencies=16
    )
    output_hybrid = encoder_hybrid(coords)
    expected_dim = 2 + (2 * 16)  # raw coords + Fourier
    assert output_hybrid.shape == (batch_size, expected_dim), \
        f"Expected shape (10, {expected_dim}), got {output_hybrid.shape}"
    print(f"   ✓ Output shape: {output_hybrid.shape}")

    print("\n✓ All BasisFunctionEncoder tests passed!")


def test_gedi_neural_process():
    """Test GEDINeuralProcess with different basis function types."""
    print("\n" + "="*60)
    print("Testing GEDINeuralProcess integration...")

    # Test parameters
    n_context = 20
    n_query = 10
    patch_size = 3

    # Create dummy data
    context_coords = torch.randn(n_context, 2)
    context_embeddings = torch.randn(n_context, patch_size, patch_size, 128)
    context_agbd = torch.randn(n_context, 1)
    query_coords = torch.randn(n_query, 2)
    query_embeddings = torch.randn(n_query, patch_size, patch_size, 128)

    basis_types = ['none', 'fourier_random', 'fourier_learnable', 'hybrid']

    for basis_type in basis_types:
        print(f"\nTesting with basis_function_type='{basis_type}'...")

        model = GEDINeuralProcess(
            patch_size=patch_size,
            embedding_channels=128,
            embedding_feature_dim=128,
            context_repr_dim=128,
            hidden_dim=256,
            latent_dim=128,
            output_uncertainty=True,
            architecture_mode='anp',
            num_attention_heads=4,
            basis_function_type=basis_type,
            basis_num_frequencies=16,
            basis_frequency_scale=1.0,
            basis_learnable=(basis_type == 'fourier_learnable')
        )

        # Forward pass
        pred_mean, pred_log_var, z_mu, z_log_sigma = model(
            context_coords,
            context_embeddings,
            context_agbd,
            query_coords,
            query_embeddings,
            training=True
        )

        # Check outputs
        assert pred_mean.shape == (n_query, 1), \
            f"Expected pred_mean shape (10, 1), got {pred_mean.shape}"
        assert pred_log_var.shape == (n_query, 1), \
            f"Expected pred_log_var shape (10, 1), got {pred_log_var.shape}"
        assert z_mu.shape == (1, 128), \
            f"Expected z_mu shape (1, 128), got {z_mu.shape}"
        assert z_log_sigma.shape == (1, 128), \
            f"Expected z_log_sigma shape (1, 128), got {z_log_sigma.shape}"

        print(f"   ✓ Forward pass successful")
        print(f"   ✓ Output shapes: pred_mean={pred_mean.shape}, "
              f"pred_log_var={pred_log_var.shape}")
        print(f"   ✓ Latent shapes: z_mu={z_mu.shape}, z_log_sigma={z_log_sigma.shape}")

        # Test backward pass
        loss = pred_mean.mean()
        loss.backward()
        print(f"   ✓ Backward pass successful")

        # Check that basis function gradients exist if learnable
        if basis_type == 'fourier_learnable':
            assert model.basis_function_encoder.frequency_matrix.grad is not None, \
                "Frequency matrix should have gradients"
            print(f"   ✓ Frequency matrix has gradients (learnable)")

    print("\n✓ All GEDINeuralProcess tests passed!")


def test_parameter_counts():
    """Compare parameter counts across different basis function types."""
    print("\n" + "="*60)
    print("Comparing parameter counts...")

    basis_configs = [
        ('none', 32, False),
        ('fourier_random', 32, False),
        ('fourier_learnable', 32, True),
        ('hybrid', 32, False),
    ]

    for basis_type, num_freq, learnable in basis_configs:
        model = GEDINeuralProcess(
            basis_function_type=basis_type,
            basis_num_frequencies=num_freq,
            basis_learnable=learnable
        )

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print(f"\n{basis_type:20s}: {trainable_params:,} trainable params")


if __name__ == '__main__':
    print("="*60)
    print("Basis Function Implementation Test Suite")
    print("="*60)

    test_basis_function_encoder()
    test_gedi_neural_process()
    test_parameter_counts()

    print("\n" + "="*60)
    print("✓ All tests passed successfully!")
    print("="*60)
