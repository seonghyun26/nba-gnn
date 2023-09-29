cd ../../

DATASET="peptides-func_lg_backtrack"
model="GCN"

python main.py \
    --repeat 3 \
    --cfg configs/ablations/$DATASET-$model.yaml \
    wandb.use False