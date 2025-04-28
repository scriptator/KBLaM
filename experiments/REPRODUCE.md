# Reproducing the experiments from the paper

The following commands allow you to reproduce the experiments from the paper on the synthetic dataset.

## Generate knowledge base embeddings
```bash
jq -c ".[]" datasets/synthetic.json > datasets/synthetic.jsonl

CUDA_VISIBLE_DEVICES=0 uv run --group experiment \
  dataset_generation/generate_kb_embeddings.py \
  --model_name all-MiniLM-L6-v2 \
  --dataset_path datasets/synthetic.jsonl \
  --dataset_name synthetic \
  --output_path ./datasets
```

## Train
```bash
CUDA_VISIBLE_DEVICES=0 uv run --group experiment \
  python experiments/train.py \
  --dataset_dir datasets/ \
  --train_dataset synthetic \
  --N 120000 \
  --B 20 \
  --lr 0.0005  \
  --use_lr_decay  \
  --use_cached_embd \
  --encoder_spec all-MiniLM-L6-v2  \
  --key_embd_src key \
  --use_data_aug \
  --hf_token hf_"eaQaJGWkXWBPiKakrlGzFwShUqTsEunHeo" \
  --hf_model_spec "meta-llama/Meta-Llama-3-8B" \
  --llm_type llama3
```


## Evaluate
```bash
CUDA_VISIBLE_DEVICES=0 uv run --group experiment \
  python experiments/eval.py \
  generation \
  --dataset_dir datasets/ \
  --test_dataset synthetic.json \
  --precomputed_embed_keys_path datasets/synthetic_all-MiniLM-L6-v2_embd_key.npy \
  --precomputed_embed_values_path datasets/synthetic_all-MiniLM-L6-v2_embd_value.npy \
  --encoder_spec all-MiniLM-L6-v2  \
  --llm_type llama3 \
  --model_dir output/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier1UseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_12000 \
  --seed 0 \
  --save_dir experiments \
  --llm_base_dir meta-llama/Meta-Llama-3-8B \
  --encoder_dir output/stage1_lr_0.0005KBTokenLayerFreq3UseOutlier1UseDataAugKeyFromkey_all-MiniLM-L6-v2_synthetic_llama3_step_12000_encoder/encoder.pt
```
