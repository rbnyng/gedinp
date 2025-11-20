# Pareto Frontier Visualization - Usage Examples

Based on your actual data showing z_std ranges from 0.7 to 188, here are the recommended visualizations:

## Quick Start (Recommended)

### 1. Basic Calibration Plot with Auto Log Scale

```bash
python plot_pareto_frontier.py \
    --input outputs_pareto/pareto_results.csv \
    --output figures/pareto_calibration.png \
    --show_pareto
```

**What you'll see:**
- Automatically enables log scale for z_std (range is 267x!)
- Separate panels for RF and XGB
- Reference line at z_std = 1.0
- Pareto frontier showing optimal configurations

**Key insights:**
- Shallow RF (depth=1): Massively overconfident (z_std ~76)
- Shallow XGB (depth=1): Nearly perfect calibration (z_std ~1.08-1.12)
- Deep trees: Increasingly overconfident as depth grows

---

### 2. Show Training Time Tradeoff (Size Encoding)

```bash
python plot_pareto_frontier.py \
    --input outputs_pareto/pareto_results.csv \
    --output figures/pareto_with_time.png \
    --show_pareto \
    --encode_time size
```

**What you'll see:**
- Larger points = slower training
- Instantly see which configs are expensive
- RF depth=20, n_estimators=1000: 311s (huge point)
- XGB depth=1, n_estimators=50: 16s (tiny point)

**Key insights:**
- Training time spans 16s to 2257s (141x range!)
- Deeper/more estimators = exponentially slower
- Can you get good enough calibration with fast configs?

---

### 3. Efficiency Frontier (Speed vs Accuracy)

```bash
python plot_pareto_frontier.py \
    --input outputs_pareto/pareto_results.csv \
    --output figures/pareto_efficiency.png \
    --plot_type efficiency \
    --show_pareto
```

**What you'll see:**
- X-axis: Training time (log scale)
- Y-axis: Test Log RMSE
- Shows diminishing returns of deeper/bigger models

**Key insights:**
- Is the 10x slowdown for depth=20 worth the RMSE improvement?
- Where does the Pareto frontier bend?
- RF vs XGB efficiency comparison

---

### 4. Generate All Visualizations at Once

```bash
python plot_pareto_frontier.py \
    --input outputs_pareto/pareto_results.csv \
    --output figures/pareto_main.png \
    --plot_type both \
    --show_pareto \
    --encode_time size
```

**Generates:**
- `pareto_main_calibration.png` - Accuracy vs calibration (with time as point size)
- `pareto_main_efficiency.png` - Accuracy vs training time
- High-res versions of both (600 dpi)

---

## Advanced Options

### Use log(z_std) Instead of z_std

```bash
python plot_pareto_frontier.py \
    --input outputs_pareto/pareto_results.csv \
    --output figures/pareto_log_z.png \
    --y_metric log_z_std \
    --show_pareto
```

**Why use this:**
- Symmetric around 0 (log(1.0) = 0)
- log(0.5) = -0.69, log(2.0) = +0.69 (symmetric under/overconfidence)
- Linear scale works well (no need for log y-axis)
- Better for extreme ranges

**Interpreting log(z_std):**
- 0 = perfect calibration
- Negative = underconfident (uncertainty too large)
- Positive = overconfident (uncertainty too small)
- ±0.69 = factor of 2 off from ideal
- ±2.3 = factor of 10 off from ideal

---

### Color Encoding for Training Time

```bash
python plot_pareto_frontier.py \
    --input outputs_pareto/pareto_results.csv \
    --output figures/pareto_color_time.png \
    --show_pareto \
    --encode_time color
```

**What you'll see:**
- Yellow = fast training
- Orange/Red = slow training
- Creates a heatmap effect
- Easier to see gradients than point size

---

### Combined Panel (All Models Together)

```bash
python plot_pareto_frontier.py \
    --input outputs_pareto/pareto_results.csv \
    --output figures/pareto_combined.png \
    --combined \
    --show_pareto
```

**Use when:**
- You want direct RF vs XGB comparison
- Creating slides/presentations
- Space-constrained publications

---

## Interpreting Your Results

Based on the data you showed:

### Random Forest Calibration Pattern
```
depth=1:  z_std ~70-76   (TERRIBLE - massively overconfident)
depth=2:  z_std ~26-41   (Still very bad)
depth=3:  z_std ~16-28   (Getting better but still bad)
depth=4:  z_std ~15-17   (Moderately bad)
depth=6:  z_std ~4-5     (Overconfident but reasonable)
depth=8:  z_std ~3-4     (Good range)
depth=10: z_std ~2-4     (Good range)
depth=20: z_std ~1.5-7   (Best calibration!)
```

**Insight:** RF needs depth ≥8 for decent calibration. Shallow RF is completely broken.

### XGBoost Calibration Pattern
```
depth=1:  z_std ~1.08-1.12   (EXCELLENT!)
depth=2:  z_std ~1.27-2.0    (Very good)
depth=3:  z_std ~1.30-1.94   (Very good)
depth=4:  z_std ~1.35-1.71   (Good)
depth=6:  z_std ~1.43-2.0    (Good)
depth=8:  z_std ~1.46-1.69   (Good)
depth=10: z_std ~1.49-2.0    (Good)
depth=20: z_std ~1.68-189    (BROKEN! Deep models completely fail)
```

**Insight:** XGB depth=1-4 gives best calibration. Very deep XGB breaks catastrophically.

### Training Time vs Calibration
```
Fast configs (<50s):  Mixed calibration (some terrible, some good)
Medium (50-200s):     Generally better calibration
Slow (>500s):         Not necessarily better calibration
```

**Insight:** Throwing compute at the problem doesn't automatically fix calibration. XGB depth=1 (16s) beats RF depth=20 (311s) on calibration.

---

## Recommended Workflow

1. **First:** Generate basic calibration plot to understand the landscape
   ```bash
   python plot_pareto_frontier.py --input outputs_pareto/pareto_results.csv \
       --output figures/pareto.png --show_pareto
   ```

2. **Second:** Look at summary statistics (printed automatically)
   - Find configs with z_std closest to 1.0
   - Find configs with best accuracy
   - Identify the tradeoff

3. **Third:** Add training time encoding
   ```bash
   python plot_pareto_frontier.py --input outputs_pareto/pareto_results.csv \
       --output figures/pareto_time.png --show_pareto --encode_time size
   ```

4. **Fourth:** Generate efficiency frontier
   ```bash
   python plot_pareto_frontier.py --input outputs_pareto/pareto_results.csv \
       --output figures/efficiency.png --plot_type efficiency --show_pareto
   ```

5. **For paper:** Generate all variations and pick the clearest
   ```bash
   python plot_pareto_frontier.py --input outputs_pareto/pareto_results.csv \
       --output figures/pareto.png --plot_type both --show_pareto --encode_time size
   ```

---

## Paper Figure Recommendations

**Main Figure (calibration):**
- Use z_std on log scale with reference line
- Separate panels for RF and XGB
- Show Pareto frontier
- Encode training time as point size
- Caption: "Accuracy-calibration tradeoff. Point size indicates training time."

**Supplement Figure (efficiency):**
- Training time vs accuracy
- Shows diminishing returns of deeper models
- Helps justify config selection

**Key Result to Highlight:**
"XGBoost with depth=1 achieves near-perfect calibration (z_std=1.08) with only 16s training time, while Random Forest requires depth≥8 and 60s+ to achieve comparable calibration quality (z_std~3-4). Very deep models (depth=20) provide marginal accuracy gains but severely degrade calibration (XGB: z_std up to 189)."
