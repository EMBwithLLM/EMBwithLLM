epoch=8
scale=small
for dataset in banking77
do
    CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python finetune_gte.py \
        --model_name_or_path thenlper/gte-large \
        --output_dir checkpoints/finetune-pretrain-1024-noprior-compare/gte-large-${dataset}-epoch=${epoch} \
        --train_file ../converted_triplet_results/${dataset}_embed=gte_s=${scale}_m=1024_d=77.0_sf_choice_seed=100-starling-train.json \
        --cache_dir cache \
        --max_source_length 512 \
        --num_train_epochs $epoch \
        --per_device_train_batch_size 2 \
        --learning_rate 5e-7 \
        --save_steps 1280 \
        --cl_temperature 0.1
done

