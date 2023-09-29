cd ../

# Datasets: peptides-func_lg_bb, peptides-func_lg, peptides-struct_lg_bb, peptides-struct_lg, vocsuperpixels_lg
# Model: GCN, GINE, GatedGCN
DATASET="peptides-func_lg_bb"
MODEL="GCN"
CONFIGPATH="configs/NBA/best/"

python main.py \
    --repeat 3 \
    --cfg c$CONFIGPATH$DATASET-$MODEL.yaml \
    wandb.use False