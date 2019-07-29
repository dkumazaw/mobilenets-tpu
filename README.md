# MobileNetV3 TPU Estimator implementation
MobileNetV3 tf.keras implementation with pre-trained weights using the TPU Estimator API.

## Requirements
I have tested this implementation using tensorflow 1.13


## ImageNet pre-trained weights
|         	| Top-1 Acc. 	| Top-5 Acc. 	| Path to weights: 	|
|:-------:	|:----------:	|:----------:	|:----------------:	|
| V3Large 	|    75.0%   	|    92.0%   	|  [Google drive](https://drive.google.com/drive/folders/1SuH_lqazrqqRBClBaIZ8BPPf4JSJgjvt?usp=sharing)  |
| V3Small 	|    67.3%   	|    87.4%   	|  [Google drive](https://drive.google.com/drive/folders/14yX5aO6oMuwHuUChUK_YcXHQNrbF8E6m?usp=sharing)  |
## Train on ImageNet
Please see `main.py` for detailed information on available flags.
```
export TPU_NAME=<your TPU name>
export MODEL_NAME=MobileNetV3Small 
export STORAGE_BUCKET=<your imagenet bucket location>
export DATA_DIR=${STORAGE_BUCKET}
export OUTPUT_DIR=${STORAGE_BUCKET}/mobilenet-test
python3 main.py --tpu=$TPU_NAME --data_dir=$DATA_DIR --model_name=$MODEL_NAME \
                --model_dir=$OUTPUT_DIR --export_dir=$OUTPUT_DIR/export 
```

## Acknowledgement
This repo's implementation is built upon the codebase of [MnasNet's TPU implementation](https://github.com/tensorflow/tpu/tree/master/models/official/mnasnet) as well as that of [Single-path NAS](https://github.com/dstamoulis/single-path-nas).
