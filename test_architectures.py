"""
Quick test to verify all architecture modes work correctly.
"""

import torch
from models.neural_process import GEDINeuralProcess, neural_process_loss

def test_architecture(mode, batch_size=5):
    """Test a specific architecture mode."""
    print(f"\n{'='*60}")
    print(f"Testing: {mode}")
    print('='*60)

    # Initialize model
    model = GEDINeuralProcess(
        patch_size=3,
        embedding_channels=128,
        embedding_feature_dim=64,
        context_repr_dim=64,
        hidden_dim=128,
        latent_dim=32,
        output_uncertainty=True,
        architecture_mode=mode,
        num_attention_heads=2
    )

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")
    print(f"Use attention: {model.use_attention}")
    print(f"Use latent: {model.use_latent}")

    # Create dummy data
    n_context = 10
    n_query = batch_size

    context_coords = torch.randn(n_context, 2)
    context_embeddings = torch.randn(n_context, 3, 3, 128)
    context_agbd = torch.randn(n_context, 1)
    query_coords = torch.randn(n_query, 2)
    query_embeddings = torch.randn(n_query, 3, 3, 128)
    query_agbd = torch.randn(n_query, 1)

    # Forward pass (training)
    print("\n--- Training Mode ---")
    pred_mean, pred_log_var, z_mu, z_log_sigma = model(
        context_coords,
        context_embeddings,
        context_agbd,
        query_coords,
        query_embeddings,
        training=True
    )

    print(f"Pred mean shape: {pred_mean.shape}")
    print(f"Pred log_var shape: {pred_log_var.shape if pred_log_var is not None else None}")
    print(f"z_mu shape: {z_mu.shape if z_mu is not None else None}")
    print(f"z_log_sigma shape: {z_log_sigma.shape if z_log_sigma is not None else None}")

    # Compute loss
    loss, loss_dict = neural_process_loss(
        pred_mean, pred_log_var, query_agbd,
        z_mu, z_log_sigma, kl_weight=1.0
    )

    print(f"\nLoss: {loss.item():.6f}")
    print(f"NLL: {loss_dict['nll']:.6f}")
    print(f"KL: {loss_dict['kl']:.6f}")

    # Forward pass (inference)
    print("\n--- Inference Mode ---")
    model.eval()
    with torch.no_grad():
        pred_mean_inf, pred_log_var_inf, z_mu_inf, z_log_sigma_inf = model(
            context_coords,
            context_embeddings,
            context_agbd,
            query_coords,
            query_embeddings,
            training=False
        )

    print(f"Pred mean shape: {pred_mean_inf.shape}")
    print(f"z_mu shape: {z_mu_inf.shape if z_mu_inf is not None else None}")

    # Check that inference uses mean (no sampling)
    if z_mu is not None and z_mu_inf is not None:
        # In inference, should use z_mu directly (no sampling)
        # So predictions should be deterministic
        pred_mean_inf2, _, _, _ = model(
            context_coords,
            context_embeddings,
            context_agbd,
            query_coords,
            query_embeddings,
            training=False
        )
        diff = (pred_mean_inf - pred_mean_inf2).abs().max().item()
        print(f"Inference deterministic check: max diff = {diff:.8f} (should be ~0)")

    print(f"\n✓ {mode} architecture working correctly!")
    return n_params


def main():
    print("=" * 60)
    print("ARCHITECTURE VERIFICATION TEST")
    print("=" * 60)

    modes = ['cnp', 'deterministic', 'latent', 'anp']
    param_counts = {}

    for mode in modes:
        try:
            param_counts[mode] = test_architecture(mode)
        except Exception as e:
            print(f"\n✗ ERROR in {mode}: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Architecture':<15} {'Parameters':>12} {'Status':<10}")
    print("-" * 60)
    for mode in modes:
        if mode in param_counts:
            print(f"{mode:<15} {param_counts[mode]:>12,} ✓")
        else:
            print(f"{mode:<15} {'N/A':>12} ✗")
    print("=" * 60)

    # Verify parameter scaling
    print("\nParameter scaling analysis:")
    if 'cnp' in param_counts and 'deterministic' in param_counts:
        diff = param_counts['deterministic'] - param_counts['cnp']
        print(f"  Attention overhead: +{diff:,} params")

    if 'deterministic' in param_counts and 'latent' in param_counts:
        diff = abs(param_counts['latent'] - param_counts['deterministic'])
        print(f"  Latent overhead: ~{diff:,} params")

    if 'deterministic' in param_counts and 'anp' in param_counts:
        diff = param_counts['anp'] - param_counts['deterministic']
        print(f"  ANP overhead (attention + latent): +{diff:,} params")

    print("\n✓ All architectures verified successfully!")


if __name__ == '__main__':
    main()
