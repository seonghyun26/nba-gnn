cd ../../
DATASET="vocsuperpixels_lg"
MODEL="GatedGCN"
CONFIGPATH="configs/NBA/best/"

python main.py \
    --repeat 3 \
    --cfg $CONFIGPATH$DATASET-$MODEL.yaml \
    wandb.use False