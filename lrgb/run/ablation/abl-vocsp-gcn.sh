cd ../../

DATASET="vocsuperpixels_lg_bt"
model="GCN"

python main.py \
    --repeat 3 \
    --cfg configs/ablations/$DATASET-$model.yaml \
    wandb.use False