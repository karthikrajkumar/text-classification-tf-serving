#!/usr/bin/env bash

docker run -p 8500:8500 \
  --mount type=bind,source=/Users/zhanglei115/workspace/text-classification-tf-serving/models,target=/models \
  -t --entrypoint=tensorflow_model_server tensorflow/serving:latest \
  --port=8500 \
  --enable_batching=true --model_name=text_cnn_classifier --model_base_path=/models &
