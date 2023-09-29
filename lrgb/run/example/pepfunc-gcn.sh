cd ../../
DATASET="peptides-func_lg_bb"
MODEL="GCN"
CONFIGPATH="configs/NBA/best/"

python main.py \
    --repeat 3 \
    --cfg $CONFIGPATH$DATASET-$MODEL.yaml \
    wandb.use False