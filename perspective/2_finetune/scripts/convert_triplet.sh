scale=small
dataset=banking77
python convert_triplet.py \
    --dataset $dataset \
    --pred_path ../1_predict_triplet/predicted_triplet_results/${dataset}_embed=gte_s=${scale}_m=1024_d=77.0_sf_choice_seed=100-starling-pred.json \
    --output_path converted_triplet_results \
    --data_path ../../datasets/${dataset}/${scale}.jsonl
