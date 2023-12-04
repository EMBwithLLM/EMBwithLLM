#===== with checkpoint =====
dataset=banking77
scale=small
checkpoint_path=CHECKPOINT_PATH

CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python get_embedding_gte.py \
    --model-name-or-path thenlper/gte-large \
    --checkpoint $checkpoint_path \
    --input_path ../../datasets/${dataset}/${scale}.jsonl \
    --output_path ../../after_measures/${dataset}_${scale}_embeds_gte_after_finetuned_.hdf5 \
    --scale $scale \
    --measure