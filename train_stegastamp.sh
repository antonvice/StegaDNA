# Launch StegaStamp-style robust training (V5)
# This includes the new Validation split and Early Stopping logic.

uv run src/train.py \
    --batch_size 24 \
    --group "stegastamp-v5" \
    --run_name "v5-stegastamp-discovery" \
    --tag v5-stegastamp \
    --use_v3_noise \
    --mixed_precision \
    --lr 1e-4 \
    --lambda_bits 50.0 \
    --lambda_perceptual 2.0 \
    --epochs 500
