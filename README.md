# text-classification-tf-serving
Serving text classification model by tensorflow serving.

### 1. Train the text classification model
```bash
python train.py
```

Here we use TextCNN model as the example. More detail of the model structure and the training process can refer to another project in my github: [[link]](https://github.com/zlxy9892/text-classification-by-cnn) 

### 2. Export model
After training the model, we can choose a checkpoint file for serving.
Copy the checkpoint file to the folder ```./data/trained_models```. Then run the following script in the terminal.
```bash
python export_model.py
```
You can specify the export path and the version of the model in the python file ```export_model.py```.

### 3. Run tensorflow serving in docker
Now you can put the exported model in docker and start tensorflow serving.
You can select a version of tensorflow serving docker to download on docker hub website: [tensorflow/serving](https://hub.docker.com/r/tensorflow/serving)

For example:
```bash
docker pull tensorflow/serving:latest
```
 
Then you can start serving in docker: 
```bash
docker run -p 8500:8500 \
  --mount type=bind,source=<PATH_OF_EXPORTED_MODEL>,target=/models \
  -t --entrypoint=tensorflow_model_server tensorflow/serving:latest \
  --port=8500 \
  --enable_batching=true --model_name=text_cnn_classifier --model_base_path=/models &
```

### 4. Start tornado service
Tensorflow serving only accept the specific format of input data that is suitable for the model.
However, most time we need to do some preprocessing on the original data before feeding into the model.
Here we use [tornado](http://www.tornadoweb.org) (a python web framework and asynchronous networking library) to build a service that can transform the original input data into the format that suitable for the model and invoke tensorflow serving to return the prediction result for the client.

Run the following to start the service:
```bash
python service.py
```

<P.S. you can run the file ```run_serving.sh``` to finish step 2~4 together.>

### 5. Use a client to check the service
I wrote a simple client python file: ```client.py```, you can run this to check the tensorflow serving.
You can also use [postman](https://www.getpostman.com/) to check it. 

*Request*:
```text
POST
localhost:9898/text_classification
Body:
{"text": "nice buying experience."}
```

*Response*:
```text
{"code": 0, "prediction": 1}
```

### Reference
[1]. [https://www.tensorflow.org/](https://www.tensorflow.org/)

[2]. [https://github.com/tensorflow/serving](https://github.com/tensorflow/serving)

[3]. [https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/serving_basic.md](https://github.com/tensorflow/serving/blob/master/tensorflow_serving/g3doc/serving_basic.md)