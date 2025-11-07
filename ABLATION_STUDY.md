# Neural Process Architecture Ablation Study

## Overview

This document describes the implementation of a **true Attentive Neural Process (ANP)** architecture and ablation study for the GEDI AGB interpolation task.

## Background

The original implementation was a **deterministic attention-based spatial interpolator**, not a true Neural Process. It lacked:

1. **Stochastic latent path** - Global context representation via latent variable
2. **KL divergence loss** - Regularization for latent distribution
3. **Proper NP training** - Beta-VAE style KL warmup

This ablation study compares different architectural variants to determine which components provide value for spatial interpolation.

## Architecture Variants

### 1. CNP (Conditional Neural Process) - Baseline
- **Deterministic path**: Mean pooling of context
- **Latent path**: None
- **Characteristics**:
  - Simplest architecture
  - Context-independent aggregation
  - No query-specific attention

### 2. Deterministic - Original Implementation
- **Deterministic path**: Cross-attention from query to context
- **Latent path**: None
- **Characteristics**:
  - Query-dependent context aggregation
  - No global stochastic representation
  - Most similar to Set Transformer

### 3. Latent - Stochastic Only
- **Deterministic path**: None (or mean pooling)
- **Latent path**: Global latent variable z ~ N(μ, σ²)
- **Characteristics**:
  - Captures global context patterns
  - Context-independent aggregation
  - Epistemic uncertainty from latent sampling

### 4. ANP (Attentive Neural Process) - Full Model
- **Deterministic path**: Cross-attention aggregation
- **Latent path**: Global latent variable
- **Characteristics**:
  - Combines query-specific attention + global context
  - Both aleatoric (heteroscedastic) and epistemic (latent) uncertainty
  - Most complex, highest parameter count

## Implementation Details

### Latent Encoder

```python
class LatentEncoder(nn.Module):
    """Encode context into latent distribution N(μ, σ²)"""

    def forward(self, context_repr):
        # Mean pool context
        pooled = context_repr.mean(dim=0, keepdim=True)

        # Encode to latent distribution
        hidden = self.fc(pooled)
        mu = self.mu_head(hidden)
        log_sigma = self.log_sigma_head(hidden)

        return mu, log_sigma
```

### Loss Function

```python
def neural_process_loss(pred_mean, pred_log_var, target,
                        z_mu, z_log_sigma, kl_weight):
    # Reconstruction loss (Gaussian NLL)
    nll = 0.5 * (pred_log_var + exp(-pred_log_var) * (target - pred_mean)²)

    # KL divergence (regularization)
    kl = 0.5 * sum(exp(2*log_sigma) + mu² - 1 - 2*log_sigma)

    # Total loss with beta-VAE style weighting
    total = nll + kl_weight * kl

    return total
```

### KL Weight Scheduling

```python
# Linear warmup from 0 to kl_weight_max over kl_warmup_epochs
kl_weight = min(1.0, epoch / kl_warmup_epochs) * kl_weight_max
```

This prevents KL collapse (latent variable being ignored) during training.

## Running the Ablation Study

### Option 1: Automated Ablation Study

Run all variants automatically with fair comparison:

```bash
python run_ablation_study.py \
    --region_bbox -73 -13 -69 -9 \
    --start_time 2019-01-01 \
    --end_time 2023-12-31 \
    --embedding_year 2024 \
    --cache_dir ./cache \
    --epochs 100 \
    --batch_size 16 \
    --lr 5e-4 \
    --output_dir ./ablation_study \
    --architectures cnp deterministic latent anp
```

This will:
1. Train all 4 architecture variants
2. Use identical hyperparameters for fair comparison
3. Generate comparison plots and summary report
4. Save results to `./ablation_study/`

### Option 2: Manual Training

Train individual variants:

```bash
# CNP baseline
python train.py \
    --architecture_mode cnp \
    --region_bbox -73 -13 -69 -9 \
    --output_dir ./outputs/cnp

# Deterministic (original)
python train.py \
    --architecture_mode deterministic \
    --region_bbox -73 -13 -69 -9 \
    --output_dir ./outputs/deterministic

# Latent only
python train.py \
    --architecture_mode latent \
    --region_bbox -73 -13 -69 -9 \
    --kl_warmup_epochs 10 \
    --kl_weight_max 1.0 \
    --output_dir ./outputs/latent

# Full ANP
python train.py \
    --architecture_mode anp \
    --region_bbox -73 -13 -69 -9 \
    --kl_warmup_epochs 10 \
    --kl_weight_max 1.0 \
    --output_dir ./outputs/anp
```

### Evaluating Trained Models

```bash
python evaluate.py \
    --model_dir ./outputs/anp \
    --checkpoint best_r2_model.pt
```

This generates:
- Test set metrics (R², RMSE, MAE)
- Prediction vs truth scatter plot
- Residual analysis
- Uncertainty calibration plot

## Expected Results

### Hypothesis 1: Attention > Mean Pooling
**Deterministic vs CNP**

We expect deterministic attention to outperform mean pooling because:
- Query-specific aggregation adapts to local spatial patterns
- Attention weights nearby context shots more heavily
- More expressive than simple averaging

### Hypothesis 2: Latent Path May Not Help
**ANP vs Deterministic**

The latent path may provide **minimal improvement** because:
- GEDI interpolation is deterministic (given context, one best prediction)
- No inherent ambiguity or multi-modality
- Attention already captures spatial dependencies

**However**, latent path could help if:
- It captures region-level patterns (e.g., biome types)
- Provides better epistemic uncertainty estimates
- Improves generalization to sparse context

### Hypothesis 3: Latent Alone < Attention Alone
**Latent vs Deterministic**

We expect latent-only to underperform because:
- Global context loses spatial locality information
- No query-specific adaptation
- Effectively reduces to mean pooling + noise

## Interpreting Results

### If Deterministic Wins:
- **Conclusion**: Query-specific attention is key, latent path unnecessary
- **Recommendation**: Use deterministic architecture (fewer parameters, faster)
- **Paper position**: "Attention-based Set Function for Spatial Interpolation"

### If ANP Wins:
- **Conclusion**: Both paths provide complementary value
- **Recommendation**: Use full ANP despite complexity
- **Paper position**: "True Attentive Neural Process for GEDI"

### If Latent Wins (Unlikely):
- **Conclusion**: Global context more important than local attention
- **Recommendation**: Investigate why (data characteristics?)
- **Paper position**: "Latent Neural Process for GEDI"

## Key Metrics to Compare

1. **R² Score** - Primary metric (higher is better)
2. **RMSE** - Prediction accuracy (lower is better)
3. **MAE** - Robustness to outliers (lower is better)
4. **Mean Uncertainty** - Calibration quality
5. **Training Time** - Practical efficiency
6. **Parameter Count** - Model complexity

## Uncertainty Analysis

### Aleatoric Uncertainty (Heteroscedastic)
- Captured by `pred_log_var` in decoder
- Represents irreducible noise in data
- Present in ALL variants

### Epistemic Uncertainty (Latent)
- Captured by sampling from latent distribution
- Represents model uncertainty
- Only in `latent` and `anp` variants

### Calibration Check
Plot predicted uncertainty vs actual error:
- Well-calibrated: Points fall on diagonal
- Under-confident: Points below diagonal
- Over-confident: Points above diagonal

## Architecture Summary Table

| Variant | Attention | Latent | Parameters | Use Case |
|---------|-----------|--------|------------|----------|
| CNP | ❌ | ❌ | ~500K | Baseline |
| Deterministic | ✅ | ❌ | ~650K | Current best |
| Latent | ❌ | ✅ | ~600K | Research |
| ANP | ✅ | ✅ | ~750K | Full model |

## Citation

If you use this implementation, please cite both the Neural Process papers and clarify which variant you used:

**For Deterministic:**
> "We use an attention-based set function inspired by Attentive Neural Processes (Kim et al., 2019), but without the latent path, as we found it unnecessary for deterministic spatial interpolation."

**For Full ANP:**
> "We implement a full Attentive Neural Process (Kim et al., 2019) with both deterministic attention and stochastic latent paths."

## References

1. **CNP**: Garnelo et al., "Conditional Neural Processes", ICML 2018
2. **ANP**: Kim et al., "Attentive Neural Processes", ICLR 2019
3. **Beta-VAE**: Higgins et al., "beta-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework", ICLR 2017

## File Structure

```
gedinp/
├── models/
│   └── neural_process.py          # Updated with all variants
├── train.py                        # Training script with architecture_mode
├── evaluate.py                     # Evaluation script
├── run_ablation_study.py          # Automated ablation experiments
├── ABLATION_STUDY.md              # This document
└── outputs/
    ├── cnp/                       # CNP results
    ├── deterministic/             # Deterministic results
    ├── latent/                    # Latent results
    └── anp/                       # ANP results
```

## Next Steps

1. **Run ablation study** on your GEDI region
2. **Analyze results** to determine best architecture
3. **Update paper** with accurate architectural description
4. **Report findings** in ablation study section
5. **Compare** with baselines (MLP, XGBoost, RF)

## Questions?

If uncertain about results:
- Check KL divergence values (should be > 0 for latent variants)
- Verify KL warmup is working (gradual increase)
- Compare training curves across variants
- Check uncertainty calibration plots
- Analyze per-tile performance (not just global metrics)
