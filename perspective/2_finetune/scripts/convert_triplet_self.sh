scale=small
dataset=banking77
python convert_triplet_self.py \
    --dataset $dataset \
    --pred_path ../1_predict_triplet/predicted_triplet_results/${dataset}_embed=gte_s=${scale}_m=1024_d=77.0_sf_choice_seed=100-starling-pred.json \
    --output_path converted_triplet_results \
    --feat_path ../../measures/${dataset}_${scale}_embeds_gte.hdf5 \
    --data_path ../../datasets/${dataset}/${scale}.jsonl