"""
Test script for variance calibration in baseline models.

Creates synthetic data to verify that temperature scaling works correctly.
"""

import numpy as np
from baselines.models import RandomForestBaseline, XGBoostBaseline


def test_rf_calibration():
    """Test Random Forest calibration with synthetic data."""
    print("=" * 80)
    print("Testing Random Forest Calibration")
    print("=" * 80)

    # Create synthetic training data
    np.random.seed(42)
    n_train = 200
    n_val = 50

    # Features: coordinates + dummy embeddings
    train_coords = np.random.rand(n_train, 2) * 10
    train_embeddings = np.random.randn(n_train, 3, 3, 4)  # Dummy embeddings
    # Target: simple linear function + noise
    train_agbd = 2.0 * train_coords[:, 0] + 3.0 * train_coords[:, 1] + np.random.randn(n_train) * 0.5

    # Validation data
    val_coords = np.random.rand(n_val, 2) * 10
    val_embeddings = np.random.randn(n_val, 3, 3, 4)
    val_agbd = 2.0 * val_coords[:, 0] + 3.0 * val_coords[:, 1] + np.random.randn(n_val) * 0.5

    # Train model
    print("\nTraining Random Forest...")
    model = RandomForestBaseline(n_estimators=50, max_depth=10, random_state=42)
    model.fit(train_coords, train_embeddings, train_agbd)
    print(f"Initial temperature: {model.temperature}")

    # Get uncalibrated predictions
    pred_uncal, std_uncal = model.predict(val_coords, val_embeddings, return_std=True)
    print(f"\nUncalibrated std range: [{std_uncal.min():.4f}, {std_uncal.max():.4f}]")
    print(f"Mean std: {std_uncal.mean():.4f}")

    # Calibrate
    print("\nCalibrating on validation set...")
    temperature = model.calibrate(val_coords, val_embeddings, val_agbd)
    print(f"Optimized temperature: {temperature:.4f}")

    # Get calibrated predictions
    pred_cal, std_cal = model.predict(val_coords, val_embeddings, return_std=True)
    print(f"\nCalibrated std range: [{std_cal.min():.4f}, {std_cal.max():.4f}]")
    print(f"Mean std: {std_cal.mean():.4f}")

    # Verify temperature was applied
    expected_std = std_uncal * temperature
    assert np.allclose(std_cal, expected_std), "Temperature scaling not applied correctly!"
    print("\n✓ Temperature scaling verified!")

    # Check calibration quality (z-scores should have std close to 1)
    z_scores = (val_agbd - pred_cal) / (std_cal + 1e-10)
    print(f"\nZ-score statistics:")
    print(f"  Mean: {z_scores.mean():.4f} (ideal: 0.0)")
    print(f"  Std: {z_scores.std():.4f} (ideal: 1.0)")

    print("\n✓ Random Forest calibration test passed!")
    return True


def test_xgb_calibration():
    """Test XGBoost calibration with synthetic data."""
    print("\n" + "=" * 80)
    print("Testing XGBoost Calibration")
    print("=" * 80)

    # Create synthetic training data
    np.random.seed(42)
    n_train = 200
    n_val = 50

    # Features: coordinates + dummy embeddings
    train_coords = np.random.rand(n_train, 2) * 10
    train_embeddings = np.random.randn(n_train, 3, 3, 4)
    train_agbd = 2.0 * train_coords[:, 0] + 3.0 * train_coords[:, 1] + np.random.randn(n_train) * 0.5

    # Validation data
    val_coords = np.random.rand(n_val, 2) * 10
    val_embeddings = np.random.randn(n_val, 3, 3, 4)
    val_agbd = 2.0 * val_coords[:, 0] + 3.0 * val_coords[:, 1] + np.random.randn(n_val) * 0.5

    # Train model
    print("\nTraining XGBoost...")
    model = XGBoostBaseline(n_estimators=50, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(train_coords, train_embeddings, train_agbd, fit_quantiles=True)
    print(f"Initial temperature: {model.temperature}")

    # Get uncalibrated predictions
    pred_uncal, std_uncal = model.predict(val_coords, val_embeddings, return_std=True)
    print(f"\nUncalibrated std range: [{std_uncal.min():.4f}, {std_uncal.max():.4f}]")
    print(f"Mean std: {std_uncal.mean():.4f}")

    # Calibrate
    print("\nCalibrating on validation set...")
    temperature = model.calibrate(val_coords, val_embeddings, val_agbd)
    print(f"Optimized temperature: {temperature:.4f}")

    # Get calibrated predictions
    pred_cal, std_cal = model.predict(val_coords, val_embeddings, return_std=True)
    print(f"\nCalibrated std range: [{std_cal.min():.4f}, {std_cal.max():.4f}]")
    print(f"Mean std: {std_cal.mean():.4f}")

    # Verify temperature was applied
    expected_std = std_uncal * temperature
    assert np.allclose(std_cal, expected_std), "Temperature scaling not applied correctly!"
    print("\n✓ Temperature scaling verified!")

    # Check calibration quality
    z_scores = (val_agbd - pred_cal) / (std_cal + 1e-10)
    print(f"\nZ-score statistics:")
    print(f"  Mean: {z_scores.mean():.4f} (ideal: 0.0)")
    print(f"  Std: {z_scores.std():.4f} (ideal: 1.0)")

    print("\n✓ XGBoost calibration test passed!")
    return True


if __name__ == '__main__':
    print("Variance Calibration Test Suite")
    print("=" * 80)
    print()

    try:
        test_rf_calibration()
        test_xgb_calibration()

        print("\n" + "=" * 80)
        print("ALL TESTS PASSED ✓")
        print("=" * 80)

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
