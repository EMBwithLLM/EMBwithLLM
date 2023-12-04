# ===== original embedding =====
for dataset in banking77
do
    for scale in small
    do
        CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python get_embedding_gte.py \
            --model-name-or-path thenlper/gte-large \
            --input_path ../../datasets/${dataset}/${scale}.jsonl \
            --output_path ../../measures/${dataset}_${scale}_embeds_gte.hdf5 \
            --scale $scale \
            --measure
    done
done