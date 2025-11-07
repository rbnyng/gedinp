"""
Test script to verify distance encoding and Fourier features work correctly.
"""
import torch
from models.neural_process import GEDINeuralProcess

def test_baseline_model():
    """Test original model without new features."""
    print("Testing baseline model (no Fourier, no distance bias)...")

    model = GEDINeuralProcess(
        use_fourier_encoding=False,
        use_distance_bias=False
    )

    # Create dummy data
    n_context, n_query = 10, 5
    context_coords = torch.rand(n_context, 2)
    query_coords = torch.rand(n_query, 2)
    context_embeddings = torch.rand(n_context, 3, 3, 128)
    query_embeddings = torch.rand(n_query, 3, 3, 128)
    context_agbd = torch.rand(n_context, 1)

    # Forward pass
    pred_mean, pred_log_var = model(
        context_coords, context_embeddings, context_agbd,
        query_coords, query_embeddings
    )

    assert pred_mean.shape == (n_query, 1), f"Expected shape {(n_query, 1)}, got {pred_mean.shape}"
    assert pred_log_var.shape == (n_query, 1), f"Expected shape {(n_query, 1)}, got {pred_log_var.shape}"
    print("✓ Baseline model works correctly")
    return True


def test_fourier_encoding():
    """Test model with Fourier encoding."""
    print("\nTesting Fourier encoding...")

    model = GEDINeuralProcess(
        use_fourier_encoding=True,
        fourier_frequencies=10,
        use_distance_bias=False
    )

    # Create dummy data
    n_context, n_query = 10, 5
    context_coords = torch.rand(n_context, 2)
    query_coords = torch.rand(n_query, 2)
    context_embeddings = torch.rand(n_context, 3, 3, 128)
    query_embeddings = torch.rand(n_query, 3, 3, 128)
    context_agbd = torch.rand(n_context, 1)

    # Forward pass
    pred_mean, pred_log_var = model(
        context_coords, context_embeddings, context_agbd,
        query_coords, query_embeddings
    )

    assert pred_mean.shape == (n_query, 1), f"Expected shape {(n_query, 1)}, got {pred_mean.shape}"
    assert pred_log_var.shape == (n_query, 1), f"Expected shape {(n_query, 1)}, got {pred_log_var.shape}"
    print("✓ Fourier encoding works correctly")
    return True


def test_distance_bias():
    """Test model with distance-biased attention."""
    print("\nTesting distance-biased attention...")

    model = GEDINeuralProcess(
        use_fourier_encoding=False,
        use_distance_bias=True,
        distance_bias_scale=1.0
    )

    # Create dummy data
    n_context, n_query = 10, 5
    context_coords = torch.rand(n_context, 2)
    query_coords = torch.rand(n_query, 2)
    context_embeddings = torch.rand(n_context, 3, 3, 128)
    query_embeddings = torch.rand(n_query, 3, 3, 128)
    context_agbd = torch.rand(n_context, 1)

    # Forward pass
    pred_mean, pred_log_var = model(
        context_coords, context_embeddings, context_agbd,
        query_coords, query_embeddings
    )

    assert pred_mean.shape == (n_query, 1), f"Expected shape {(n_query, 1)}, got {pred_mean.shape}"
    assert pred_log_var.shape == (n_query, 1), f"Expected shape {(n_query, 1)}, got {pred_log_var.shape}"
    print("✓ Distance-biased attention works correctly")
    return True


def test_full_features():
    """Test model with both Fourier encoding and distance bias."""
    print("\nTesting full feature set (Fourier + distance bias)...")

    model = GEDINeuralProcess(
        use_fourier_encoding=True,
        fourier_frequencies=10,
        use_distance_bias=True,
        distance_bias_scale=1.0
    )

    # Create dummy data
    n_context, n_query = 10, 5
    context_coords = torch.rand(n_context, 2)
    query_coords = torch.rand(n_query, 2)
    context_embeddings = torch.rand(n_context, 3, 3, 128)
    query_embeddings = torch.rand(n_query, 3, 3, 128)
    context_agbd = torch.rand(n_context, 1)

    # Forward pass
    pred_mean, pred_log_var = model(
        context_coords, context_embeddings, context_agbd,
        query_coords, query_embeddings
    )

    assert pred_mean.shape == (n_query, 1), f"Expected shape {(n_query, 1)}, got {pred_mean.shape}"
    assert pred_log_var.shape == (n_query, 1), f"Expected shape {(n_query, 1)}, got {pred_log_var.shape}"
    print("✓ Full feature set works correctly")
    return True


def test_gradient_flow():
    """Test that gradients flow correctly through new components."""
    print("\nTesting gradient flow...")

    model = GEDINeuralProcess(
        use_fourier_encoding=True,
        fourier_frequencies=10,
        use_distance_bias=True,
        distance_bias_scale=1.0
    )

    # Create dummy data
    n_context, n_query = 10, 5
    context_coords = torch.rand(n_context, 2)
    query_coords = torch.rand(n_query, 2)
    context_embeddings = torch.rand(n_context, 3, 3, 128)
    query_embeddings = torch.rand(n_query, 3, 3, 128)
    context_agbd = torch.rand(n_context, 1)
    target = torch.rand(n_query, 1)

    # Forward pass
    pred_mean, pred_log_var = model(
        context_coords, context_embeddings, context_agbd,
        query_coords, query_embeddings
    )

    # Compute loss and backprop
    loss = ((pred_mean - target) ** 2).mean()
    loss.backward()

    # Check that distance bias parameters have gradients
    if hasattr(model.attention_aggregator, 'log_distance_scale'):
        assert model.attention_aggregator.log_distance_scale.grad is not None, \
            "Distance scale parameter has no gradient"
        assert model.attention_aggregator.distance_bias_per_head.grad is not None, \
            "Distance bias per-head parameters have no gradient"
        print("✓ Gradients flow to distance bias parameters")

    print("✓ Gradient flow is working correctly")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Distance Encoding and Fourier Features")
    print("=" * 60)

    try:
        test_baseline_model()
        test_fourier_encoding()
        test_distance_bias()
        test_full_features()
        test_gradient_flow()

        print("\n" + "=" * 60)
        print("✓ All tests passed!")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
