#!/bin/bash

export OMP_NUM_THREADS=1

# Adjust env variables to maintain the global batch size
#
#    NGPU x BS x GRAD_ACC = 256.

[ ! -n "$OUTPUT_DIR" ] && OUTPUT_DIR="./output"
[ ! -n "$NGPU" ] && NGPU=1
[ ! -n "$BS" ] && BS=32
[ ! -n "$GRAD_ACC" ] && GRAD_ACC=8
[ ! -n "$EPOCHS" ] && EPOCHS=1500
[ "$AMP" == "true" ] && AMP_FLAG="--amp"

GBS=$(($NGPU * $BS * $GRAD_ACC))
[ $GBS -ne 256 ] && echo -e "\nWARNING: Global batch size changed from 256 to ${GBS}.\n"

echo -e "\nSetup: ${NGPU}x${BS}x${GRAD_ACC} - global batch size ${GBS}\n"

mkdir -p "$OUTPUT_DIR"
python3 train.py \
    -o "$OUTPUT_DIR/" \
    --log-file "$OUTPUT_DIR/nvlog.json" \
    --dataset-path /Users/cschaefe/datasets/asvoice2_fastpitch \
    --training-files /Users/cschaefe/datasets/asvoice2_fastpitch/metadata_train.txt \
    --validation-files /Users/cschaefe/datasets/asvoice2_fastpitch/metadata_val.txt \
    --pitch-mean-std-file /Users/cschaefe/datasets/asvoice2_fastpitch/pitch_char_stats__metadata_train_taco.json \
    --epochs ${EPOCHS} \
    --epochs-per-checkpoint 100 \
    --warmup-steps 1000 \
    -lr 0.1 \
    -bs ${BS} \
    --optimizer lamb \
    --grad-clip-thresh 1000.0 \
    --dur-predictor-loss-scale 0.1 \
    --pitch-predictor-loss-scale 0.1 \
    --weight-decay 1e-6 \
    --gradient-accumulation-steps ${GRAD_ACC} \
    ${AMP_FLAG}
