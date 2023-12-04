scale=small
for dataset in banking77 
do
    for max_query in 1024
    do
        for embed in gte
        do
            feat_path=../../measures/${dataset}_${scale}_embeds_gte.hdf5
            python triplet_sampling.py \
                --data_path ../../datasets/${dataset}/${scale}.jsonl \
                --feat_path $feat_path \
                --dataset $dataset \
                --embed_method $embed \
                --max_query $max_query \
                --filter_first_prop 0.0 \
                --large_ent_prop 0.2 \
                --out_dir sampled_triplet_results \
                --max_distance 77 \
                --scale $scale \
                --shuffle_inds \
                --seed 100
        done
    done
done