# MobileNetV3 TPU Estimator implementation
MobileNetV3 small and large implementation using the TPU Estimator API.

## Requirements
I have tested this implementation using tensorflow 1.13

## Train on ImageNet
You can run `main.py` as follows:
```
export TPU_NAME=<your TPU name>
export MODEL_NAME=MobileNetV3Small 
export STORAGE_BUCKET=<your imagenet bucket location>
export DATA_DIR=${STORAGE_BUCKET}
export OUTPUT_DIR=${STORAGE_BUCKET}/mobilenet-test2
python3 main.py --tpu=$TPU_NAME --data_dir=$DATA_DIR --model_name=$MODEL_NAME \
                --model_dir=$OUTPUT_DIR --export_dir=$OUTPUT_DIR/export 
```

## Acknowledgement
This repo extends the codebase of [MnasNet's TPU implementation](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet).
