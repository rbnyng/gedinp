# Training Improvements for ConvCNP

## Issues Identified

### 1. **Severe Overfitting** ❌

**Symptoms:**
- Training loss drops to ~0.0000 by epoch 3
- Validation RMSE stays around 0.19-0.23 (not improving much)
- Validation R² peaks at ~0.40 (epoch 79-80) then **degrades to negative** by epoch 100
- Best model was around epoch 80, not epoch 1

**Diagnosis:**
```
Train Loss: 0.0000  ← memorizing training data
Val Loss:   0.0000  ← but still has errors (just displayed as 0.0000)
Val R²:     -0.0135 ← worse than predicting the mean!
```

This is **classic overfitting**: the model perfectly fits training data but fails to generalize.

---

### 2. **Loss Display Issue** 🔢

**Problem:**
- Losses show as `0.0000` due to Python's default float formatting
- They're actually non-zero (e.g., `1.234e-05`) but rounded

**Impact:**
- Hard to see if training loss is actually decreasing
- Can't distinguish between `1e-5` and `1e-8`

---

### 3. **No Early Stopping** ⏱️

**Problem:**
- Training runs for all 100 epochs even though best model was at epoch 80
- Wasted 20 epochs of compute time
- Final model is **worse** than the best model

---

## Solutions

### 1. Early Stopping Implementation

**File:** `train_convcnp_improved.py`

```python
class EarlyStopping:
    """Stop training when validation metric stops improving."""

    def __init__(self, patience=15, min_delta=0.01, mode='max'):
        """
        Args:
            patience: Wait 15 epochs after last improvement
            min_delta: Must improve by at least 1%
            mode: 'max' for R², 'min' for loss
        """
        ...

# In training loop:
early_stopping = EarlyStopping(patience=15, mode='max')

for epoch in range(epochs):
    ...
    if early_stopping(val_r2, epoch):
        print(f"🛑 Early stopping at epoch {epoch}")
        break
```

**Benefits:**
- Stops training automatically when validation plateaus
- Saves compute time
- Prevents degradation after peak performance

**Recommended settings for your case:**
- `patience=10-15` (wait 10-15 epochs)
- Monitor **validation R²** (maximize)
- Stop when R² doesn't improve by ≥1%

---

### 2. Scientific Notation for Losses

**Before:**
```python
print(f"Train Loss: {train_loss:.4f}")  # Shows: 0.0000
print(f"Val Loss:   {val_loss:.4f}")    # Shows: 0.0000
```

**After:**
```python
print(f"Train Loss: {train_loss:.6e}")  # Shows: 1.234567e-05
print(f"Val Loss:   {val_loss:.6e}")    # Shows: 3.456789e-04
```

**Benefits:**
- Can see actual loss values even when very small
- Can track training progress properly
- Better for debugging

---

### 3. Additional Improvements to Combat Overfitting

#### A. **Weight Decay (L2 Regularization)**

```python
# Before:
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# After:
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-5  # L2 penalty on weights
)
```

#### B. **Learning Rate Scheduling**

```python
# Reduce LR when validation plateaus
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',      # Maximize R²
    factor=0.5,      # Reduce by half
    patience=5,      # Wait 5 epochs
    min_lr=1e-7
)

# After each epoch:
scheduler.step(val_r2)
```

#### C. **Gradient Clipping**

```python
# Prevent exploding gradients
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

#### D. **Improved Loss Function**

```python
def neural_process_loss(pred_mean, pred_log_var, target, eps=1e-8):
    """NLL loss with numerical stability."""

    # Clamp log_var to prevent extreme values
    pred_log_var = torch.clamp(pred_log_var, min=-10, max=10)

    # NLL = 0.5 * (log_var + (y - μ)²/var)
    nll = 0.5 * (
        pred_log_var +
        ((pred_mean - target) ** 2) / (torch.exp(pred_log_var) + eps)
    )
    return torch.mean(nll)
```

---

## How to Use

### Option 1: Quick Fix (Minimal Changes)

Add to your existing training script:

```python
# 1. Add early stopping
from train_convcnp_improved import EarlyStopping

early_stopping = EarlyStopping(patience=15, mode='max', verbose=True)

# 2. Fix loss printing
print(f"Train Loss: {train_loss:.6e}")  # Instead of .4f
print(f"Val Loss:   {val_loss:.6e}")

# 3. Add weight decay
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.lr,
    weight_decay=1e-5
)

# 4. Check early stopping after each epoch
if early_stopping(val_r2, epoch):
    print(f"Early stopping triggered!")
    break
```

### Option 2: Use Complete Improved Training Loop

```python
from train_convcnp_improved import (
    train_with_improvements,
    create_optimizer_and_scheduler
)

# Create optimizer with weight decay + scheduler
optimizer, scheduler = create_optimizer_and_scheduler(
    model,
    lr=1e-4,
    weight_decay=1e-5
)

# Run improved training loop
history = train_with_improvements(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    optimizer=optimizer,
    scheduler=scheduler,
    device=device,
    epochs=100,
    early_stopping_patience=15,
    output_dir='./outputs_convcnp'
)
```

---

## Analysis Tools

### Visualize Training Dynamics

```bash
python analyze_training.py \
    --history outputs_convcnp/history.json \
    --save_plot training_curves.png
```

**Generates:**
1. **Loss curves** (train vs val) with log scale
2. **RMSE over time** with best epoch marker
3. **R² over time** with degradation detection
4. **Learning rate schedule**

**Plus diagnostic report:**
- Overfitting detection
- Train/val gap analysis
- Recommendations for improvement

---

## Expected Results

**With improvements, you should see:**

1. **Training Loss:** Still decreases but not to near-zero
   ```
   Epoch 1:  Train Loss: 1.234e-01
   Epoch 50: Train Loss: 5.678e-03  ← Healthy, not 0
   ```

2. **Validation R²:** Improves steadily, then plateaus
   ```
   Epoch 1:  Val R²: -0.20
   Epoch 50: Val R²:  0.45  ← Best
   Epoch 65: Val R²:  0.44  ← Plateau
   → Early stopping at epoch 65
   ```

3. **Generalization Gap:** Reduced
   ```
   Before: Val/Train ratio = 100x  (severe overfitting)
   After:  Val/Train ratio = 2-5x  (healthy)
   ```

4. **Training Time:** Reduced by ~30-50%
   ```
   Before: 100 epochs (best at 80)
   After:  50-70 epochs with early stopping
   ```

---

## Hyperparameter Recommendations

Based on your training dynamics:

```python
# Learning rate
lr = 1e-4  # Current is fine

# Weight decay (L2 regularization)
weight_decay = 1e-5  # Start here, try 1e-4 if still overfitting

# Early stopping
patience = 15         # Wait 15 epochs
min_delta = 0.01     # Must improve R² by 1%

# Scheduler
lr_patience = 5      # Reduce LR after 5 epochs of plateau
lr_factor = 0.5      # Reduce by half

# Gradient clipping
max_grad_norm = 1.0  # Clip gradients to [-1, 1]
```

---

## Quick Checklist

- [ ] Add scientific notation for loss printing (`.6e` instead of `.4f`)
- [ ] Implement early stopping with patience=15
- [ ] Add weight decay (`AdamW` with `weight_decay=1e-5`)
- [ ] Add learning rate scheduler (`ReduceLROnPlateau`)
- [ ] Add gradient clipping (`clip_grad_norm_`)
- [ ] Monitor validation R² as stopping criterion
- [ ] Run analysis script to visualize results
- [ ] Compare before/after generalization gap

---

## Questions?

The key insight: **Your model is memorizing training data instead of learning patterns.**

**Solution:** Add regularization (weight decay, early stopping, LR scheduling) to force it to learn generalizable features.

Your best model is around **epoch 79-80** with **R² ≈ 0.40**, but training continues to epoch 100 where R² becomes **negative**. This wastes compute and gives you a worse final model.

With early stopping, training would stop around epoch 90-95, giving you the best model automatically.
