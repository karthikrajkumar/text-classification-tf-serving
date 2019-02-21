# coding=utf-8

import numpy as np
import tensorflow as tf
import grpc
from tensorflow.contrib import learn
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import json
import tornado.ioloop
import tornado.web
import tornado.escape


class TextClassificationHandler(tornado.web.RequestHandler):
    """文本分类任务请求处理类"""

    def initialize(self):
        server_url = 'localhost:8500'
        channel = grpc.insecure_channel(server_url)
        self.stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
        self.tf_request = predict_pb2.PredictRequest()
        self.tf_request.model_spec.name = 'text_cnn_classifier'  # 模型名称
        self.tf_request.model_spec.signature_name = 'text_classification_tf_serving'  # 签名名称

        # load the vocabulary
        vocab_path = './data/vocab'
        self.vocab_prosessor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)

    def prepare(self):
        print('prepare...')
        data = tornado.escape.json_decode(self.request.body)
        text = data.get('text', '')
        self.input_x = self.preprocess_input(text)

    def preprocess_input(self, input_data):
        input_x = np.array(list(self.vocab_prosessor.transform([input_data]))).astype(np.int32)
        return input_x

    def get(self):
        print('GET request.')
        self.write('GET request!')

    def post(self):
        """处理POST请求，预处理输入文本，并向TensorFlow Serving服务请求模型推理结果."""
        print('POST request.')

        self.tf_request.inputs['input_x'].CopyFrom(
            tf.contrib.util.make_tensor_proto(self.input_x, shape=[1, 56]))
        self.tf_request.inputs['input_dropout_keep_prob'].CopyFrom(
            tf.contrib.util.make_tensor_proto(1.0, shape=[1]))
        tf_response = self.stub.Predict(self.tf_request, 5.0)  # 5 secs timeout
        # print(tf_response.outputs['prediction'])
        y_pred = tf_response.outputs['prediction'].int64_val[0]

        code = 0
        result = {
            "code": code,
            # "input_text": text,
            "prediction": y_pred
        }
        self.write(json.dumps(result, ensure_ascii=False))


def make_app():
    return tornado.web.Application([
        (r"/text_classification", TextClassificationHandler)
    ])


if __name__ == '__main__':
    app = make_app()
    app.listen(9898)
    tornado.ioloop.IOLoop.current().start()
