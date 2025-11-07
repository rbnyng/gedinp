# Key Code Snippets

## 1. Neural Process Forward Pass

### Main Forward Function (neural_process.py:354-414)

```python
def forward(
    self,
    context_coords: torch.Tensor,      # (n_context, 2)
    context_embeddings: torch.Tensor,  # (n_context, 3, 3, 128)
    context_agbd: torch.Tensor,        # (n_context, 1)
    query_coords: torch.Tensor,        # (n_query, 2)
    query_embeddings: torch.Tensor     # (n_query, 3, 3, 128)
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Forward pass returns:
    - pred_mean: (n_query, 1) - predicted AGBD
    - pred_log_var: (n_query, 1) - log uncertainty (or None)
    """
    
    # 1. Encode embeddings (shared encoder for context & query)
    context_emb_features = self.embedding_encoder(context_embeddings)  # (n_context, 128)
    query_emb_features = self.embedding_encoder(query_embeddings)      # (n_query, 128)
    
    # 2. Encode context points: [coords + emb_features + agbd] -> repr
    context_repr = self.context_encoder(
        context_coords,         # (n_context, 2)
        context_emb_features,   # (n_context, 128)
        context_agbd            # (n_context, 1)
    )                           # -> (n_context, 128)
    
    # 3. Aggregate context representations
    if self.use_attention:
        # Query projection: [coords + emb_features] -> context_dim
        query_repr = torch.cat([query_coords, query_emb_features], dim=-1)  # (n_query, 130)
        query_repr_projected = self.query_proj(query_repr)                  # (n_query, 128)
        
        # Cross-attention: aggregate context based on each query
        aggregated_context = self.attention_aggregator(
            query_repr_projected,  # (n_query, 128)
            context_repr           # (n_context, 128)
        )                          # -> (n_query, 128)
    else:
        # Mean pooling: simple average of all context representations
        aggregated_context = context_repr.mean(dim=0, keepdim=True)  # (1, 128)
        aggregated_context = aggregated_context.expand(query_coords.shape[0], -1)
    
    # 4. Decode: [query_coords + query_emb_features + aggregated_context] -> AGBD
    pred_mean, pred_log_var = self.decoder(
        query_coords,           # (n_query, 2)
        query_emb_features,     # (n_query, 128)
        aggregated_context      # (n_query, 128)
    )                           # -> (n_query, 1), (n_query, 1)
    
    return pred_mean, pred_log_var
```

## 2. Loss Function

### Gaussian Negative Log-Likelihood (neural_process.py:446-472)

```python
def neural_process_loss(
    pred_mean: torch.Tensor,        # (batch, 1)
    pred_log_var: Optional[torch.Tensor],  # (batch, 1) or None
    target: torch.Tensor            # (batch, 1)
) -> torch.Tensor:
    """
    Gaussian NLL loss that learns uncertainty:
    
    Loss = 0.5 * (log_var + exp(-log_var) * (target - mean)^2)
    
    This is equivalent to:
    Loss = 0.5 * (log_var + (target - mean)^2 / var)
    
    Where var = exp(log_var)
    """
    
    if pred_log_var is not None:
        # Gaussian negative log-likelihood with learned variance
        loss = 0.5 * (
            pred_log_var +
            torch.exp(-pred_log_var) * (target - pred_mean) ** 2
        )
    else:
        # Fallback to MSE if no uncertainty prediction
        loss = (target - pred_mean) ** 2
    
    return loss.mean()
```

## 3. Training Loop

### Per-Epoch Training (train.py:110-167)

```python
def train_epoch(model, dataloader, optimizer, device):
    """
    Train for one epoch.
    
    Note: dataloader returns batches of tiles, each with variable
    number of context/target points. We process each tile independently.
    """
    model.train()
    total_loss = 0
    n_tiles = 0
    
    for batch in tqdm(dataloader, desc='Training'):
        optimizer.zero_grad()
        
        batch_loss = 0
        n_tiles_in_batch = 0
        
        # Process each tile in the batch independently
        for i in range(len(batch['context_coords'])):
            context_coords = batch['context_coords'][i].to(device)
            context_embeddings = batch['context_embeddings'][i].to(device)
            context_agbd = batch['context_agbd'][i].to(device)
            target_coords = batch['target_coords'][i].to(device)
            target_embeddings = batch['target_embeddings'][i].to(device)
            target_agbd = batch['target_agbd'][i].to(device)
            
            # Skip if no target points
            if len(target_coords) == 0:
                continue
            
            # Forward pass
            pred_mean, pred_log_var = model(
                context_coords,
                context_embeddings,
                context_agbd,
                target_coords,
                target_embeddings
            )
            
            # Compute loss
            loss = neural_process_loss(pred_mean, pred_log_var, target_agbd)
            
            # Check for numerical issues
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected. Skipping tile.")
                continue
            
            batch_loss += loss
            n_tiles_in_batch += 1
        
        # Aggregate and backprop
        if n_tiles_in_batch > 0:
            batch_loss = batch_loss / n_tiles_in_batch
            batch_loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += batch_loss.item()
            n_tiles += n_tiles_in_batch
    
    return total_loss / max(n_tiles, 1)
```

## 4. Data Preparation

### Context/Target Split (dataset.py:108-172)

```python
def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
    """
    Get a training sample (one tile).
    
    For each tile:
    1. Normalize coordinates to [0, 1]
    2. Split into context and target randomly
    3. Apply data augmentation (if training)
    4. Normalize AGBD values
    """
    
    tile_data = self.tiles[idx].copy()
    n_shots = len(tile_data)
    
    # Random context/target split
    # Context ratio sampled uniformly from [0.3, 0.7]
    context_ratio = random.uniform(*self.context_ratio_range)  # e.g., 0.6
    n_context = max(1, int(n_shots * context_ratio))           # e.g., 6 context shots
    
    # Randomly select which shots are context vs target
    context_indices = random.sample(range(n_shots), n_context)
    target_indices = [i for i in range(n_shots) if i not in context_indices]
    
    # Extract raw data
    coords = tile_data[['longitude', 'latitude']].values        # (N, 2)
    embeddings = np.stack(tile_data['embedding_patch'].values)  # (N, 3, 3, 128)
    agbd = tile_data['agbd'].values[:, None]                    # (N, 1)
    
    # Normalize coordinates to [0, 1] within tile bounds
    if self.normalize_coords:
        coords = self._normalize_coordinates(coords, tile_data)
    
    # Optional: Add small random noise to coordinates during training
    if self.augment_coords:
        coords = coords + np.random.normal(0, self.coord_noise_std, coords.shape)
        coords = np.clip(coords, 0, 1)  # Keep in [0, 1]
    
    # Normalize AGBD
    if self.normalize_agbd:
        if self.log_transform_agbd:
            # Log transform: log(1 + x)
            # Then normalize by log(1 + scale) where scale=200
            agbd = np.log1p(agbd) / np.log1p(self.agbd_scale)
        else:
            agbd = agbd / self.agbd_scale
    
    # Split into context and target
    context_coords = coords[context_indices]
    context_embeddings = embeddings[context_indices]
    context_agbd = agbd[context_indices]
    
    target_coords = coords[target_indices]
    target_embeddings = embeddings[target_indices]
    target_agbd = agbd[target_indices]
    
    return {
        'context_coords': torch.from_numpy(context_coords).float(),
        'context_embeddings': torch.from_numpy(context_embeddings).float(),
        'context_agbd': torch.from_numpy(context_agbd).float(),
        'target_coords': torch.from_numpy(target_coords).float(),
        'target_embeddings': torch.from_numpy(target_embeddings).float(),
        'target_agbd': torch.from_numpy(target_agbd).float(),
    }
```

## 5. Embedding Extraction

### GeoTessera Patch Extraction (embeddings.py:159-206)

```python
def extract_patch(
    self,
    lon: float,
    lat: float
) -> Optional[np.ndarray]:
    """
    Extract a 3x3 patch of GeoTessera embeddings around a point.
    
    Args:
        lon: Longitude (WGS84)
        lat: Latitude (WGS84)
    
    Returns:
        Patch array (3, 3, 128) or None if fails
    """
    
    # Step 1: Determine which GeoTessera tile contains this point
    # Tiles are 0.1° × 0.1° centered at multiples of 0.1
    tile_lon, tile_lat = self._get_tile_coords(lon, lat)
    
    # Step 2: Load the tile (from cache or download)
    tile_data = self._load_tile(tile_lon, tile_lat)
    if tile_data is None:
        return None
    
    embedding, crs, transform = tile_data
    # embedding shape: (height, width, 128)
    
    # Step 3: Convert lon/lat to pixel coordinates within the tile
    # Tiles are in UTM or similar projected coordinates
    try:
        row, col = self._lonlat_to_pixel(lon, lat, transform, crs)
    except Exception as e:
        print(f"Warning: Could not convert coordinates ({lon}, {lat}): {e}")
        return None
    
    # Step 4: Extract 3×3 patch centered at (row, col)
    height, width, channels = embedding.shape
    half_patch = self.patch_size // 2  # 1 for patch_size=3
    
    # Check bounds
    if (row - half_patch < 0 or row + half_patch + 1 > height or
        col - half_patch < 0 or col + half_patch + 1 > width):
        return None  # Point too close to edge
    
    # Extract patch: (3, 3, 128)
    patch = embedding[
        row - half_patch : row + half_patch + 1,
        col - half_patch : col + half_patch + 1,
        :
    ]
    
    return patch
```

## 6. Spatial Cross-Validation

### Tile-Based Split (spatial_cv.py:50-81)

```python
def split(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create train/val/test split at TILE LEVEL.
    
    This ensures no spatial leakage: tiles in different splits
    are completely separate spatial regions.
    """
    
    # Get all unique tile IDs
    tile_ids = self.data_df['tile_id'].unique()
    self.n_tiles = len(tile_ids)
    
    # Shuffle tiles (for randomness)
    shuffled_tiles = tile_ids.copy()
    np.random.shuffle(shuffled_tiles)
    
    # Calculate split sizes
    n_test = max(1, int(self.n_tiles * self.test_ratio)) if self.test_ratio > 0 else 0
    n_val = max(1, int(self.n_tiles * self.val_ratio)) if self.val_ratio > 0 else 0
    n_train = self.n_tiles - n_test - n_val
    
    # Split tiles (not individual shots)
    train_tiles = shuffled_tiles[:n_train]
    val_tiles = shuffled_tiles[n_train : n_train + n_val]
    test_tiles = shuffled_tiles[n_train + n_val :]
    
    # Create dataframe splits by filtering on tile ID
    train_df = self.data_df[self.data_df['tile_id'].isin(train_tiles)]
    val_df = self.data_df[self.data_df['tile_id'].isin(val_tiles)]
    test_df = self.data_df[self.data_df['tile_id'].isin(test_tiles)]
    
    print(f"Spatial split created:")
    print(f"  Train: {len(train_tiles)} tiles, {len(train_df)} shots")
    print(f"  Val:   {len(val_tiles)} tiles, {len(val_df)} shots")
    print(f"  Test:  {len(test_tiles)} tiles, {len(test_df)} shots")
    
    return train_df, val_df, test_df
```

## 7. Metrics Computation

### Evaluation Metrics (neural_process.py:475-514)

```python
def compute_metrics(
    pred_mean: torch.Tensor,
    pred_std: Optional[torch.Tensor],
    target: torch.Tensor
) -> dict:
    """
    Compute evaluation metrics (RMSE, MAE, R², uncertainty).
    """
    
    # Convert to numpy
    pred_mean = pred_mean.detach().cpu().numpy().flatten()
    target = target.detach().cpu().numpy().flatten()
    
    # Root Mean Squared Error
    mse = ((pred_mean - target) ** 2).mean()
    rmse = mse ** 0.5
    
    # Mean Absolute Error
    mae = abs(pred_mean - target).mean()
    
    # R² Score (coefficient of determination)
    # R² = 1 - SS_res / SS_tot
    # where SS_res = sum((y_true - y_pred)²)
    #       SS_tot = sum((y_true - y_mean)²)
    ss_res = ((target - pred_mean) ** 2).sum()
    ss_tot = ((target - target.mean()) ** 2).sum()
    r2 = 1 - ss_res / (ss_tot + 1e-8)
    
    metrics = {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mse': mse
    }
    
    # Optional: Mean uncertainty (prediction standard deviation)
    if pred_std is not None:
        pred_std = pred_std.detach().cpu().numpy().flatten()
        metrics['mean_uncertainty'] = pred_std.mean()
    
    return metrics
```

## 8. Complete Training Example

```bash
# Full training command
python train.py \
    --region_bbox 30.256 -15.853 30.422 -15.625 \
    --start_time 2019-01-01 \
    --end_time 2023-12-31 \
    --embedding_year 2024 \
    --patch_size 3 \
    --hidden_dim 512 \
    --embedding_feature_dim 128 \
    --context_repr_dim 128 \
    --use_attention \
    --num_attention_heads 4 \
    --batch_size 16 \
    --lr 5e-4 \
    --epochs 100 \
    --val_ratio 0.15 \
    --test_ratio 0.15 \
    --min_shots_per_tile 10 \
    --early_stopping_patience 15 \
    --log_transform_agbd \
    --augment_coords \
    --output_dir ./outputs \
    --device cuda \
    --seed 42
```

### What Happens During Training:

1. **Data Query**: Fetches GEDI shots in region (may take minutes)
2. **Embedding Extraction**: Extracts 3×3 GeoTessera patches (~few minutes)
3. **Spatial Split**: Divides tiles into train/val/test
4. **Dataset Creation**: Loads all data into memory (pandas + numpy)
5. **Training Loop** (100 epochs):
   - Each epoch processes 16 batches (each batch = 16 tiles)
   - Per-tile: random context/target split (30-70%)
   - Forward pass → loss computation
   - Backprop + optimizer step
   - Validation every epoch
6. **Model Selection**:
   - Saves best model by val loss
   - Saves alternate best model by R²
   - Early stopping if no improvement for 15 epochs
7. **Output**:
   - `best_model.pt` - trained weights
   - `config.json` - training config
   - `history.json` - loss curves
   - `train/val/test_split.csv` - data splits

