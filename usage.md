python predict.py --checkpoint ./outputs \
                  --region -73 2.9 -72.9 3

python train.py \
  --region_bbox -73 2 -72 3 \
  --embedding_year 2022 \
  --start_time 2022-01-01 \
  --end_time 2022-12-31 \
  --cache_dir ./cache \
  --embeddings_dir ./embeddings \
  --output_dir ./outputs \
  --epochs 100 

python run_ablation_study.py \
  --region_bbox -73 2 -72 3 \
  --embedding_year 2022 \
  --start_time 2022-01-01 \
  --end_time 2022-12-31 \
  --cache_dir ./cache \
  --embeddings_dir ./embeddings \
  --buffer_size 0.1 \
  --output_dir ./outputs_ablation \
  --epochs 100

python train_baselines.py \
  --region_bbox -73 2 -72 3 \
  --embedding_year 2022 \
  --start_time 2022-01-01 \
  --end_time 2022-12-31 \
    --models xgb \
    --output_dir ./outputs_baselines \
    --cache_dir ./cache \
    --embeddings_dir ./embeddings

python evaluate.py \
  --model_dir ./outputs \
  --checkpoint best_r2_model.pt

python evaluate_temporal.py \
  --model_dir ./outputs \
  --test_years 2022 \
  --checkpoint best_r2_model.pt
  
python run_training_harness.py \
	--script train.py \
	--n_seeds 10 \
	--output_dir ./results/np_test \
	--embedding_year 2022 \
	--start_time 2022-01-01 \
	--end_time 2022-12-31 \
	--cache_dir ./cache \
	--embeddings_dir ./embeddings \
	--output_dir ./outputs_np \
	--epochs 100 \
	--region_bbox -73 2 -72 3

python run_training_harness.py \
    --script train_baselines.py \
	--embedding_year 2022 \
	--start_time 2022-01-01 \
	--end_time 2022-12-31 \
    --n_seeds 10 \
    --models rf xgb idw \
    --output_dir ./baselines \
    --region_bbox -73 2 -72 3




python train.py \
  --region_bbox -73 2 -72 3 \
  --embedding_year 2022 \
  --start_time 2022-01-01 \
  --end_time 2022-12-31 \
  --cache_dir ./cache \
  --embeddings_dir ./embeddings \
  --output_dir ./outputs \
  --epochs 100

python train_baselines.py \
  --region_bbox -73 2 -72 3 \
  --embedding_year 2022 \
  --start_time 2022-01-01 \
  --end_time 2022-12-31 \
    --models xgb \
    --output_dir ./outputs_baselines \
    --cache_dir ./cache \
    --embeddings_dir ./embeddings
