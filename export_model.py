# coding=utf-8

import os
import tensorflow as tf


def main(_):
    checkpoint_file = './data/trained_models/model-5500'
    print('Loaded the latest checkpoint: <{}>\n'.format(checkpoint_file))

    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=True)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            saver.restore(sess, checkpoint_file)

            model_version = 1
            export_path_base = './models'
            export_path = os.path.join(tf.compat.as_bytes(export_path_base), tf.compat.as_bytes(str(model_version)))

            print('Exporting trained model to <{}>'.format(export_path))
            builder = tf.saved_model.builder.SavedModelBuilder(export_path)

            # create prediction signature
            inputs = {
                'input_x': tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name('input_x').outputs[0]),
                'input_dropout_keep_prob': tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name('dropout_keep_prob').outputs[0])
            }
            outputs = {
                'prediction': tf.saved_model.utils.build_tensor_info(graph.get_operation_by_name('output/predictions').outputs[0])
            }

            signature = tf.saved_model.signature_def_utils.build_signature_def(
                inputs, outputs,
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
            builder.add_meta_graph_and_variables(
                sess, [tf.saved_model.tag_constants.SERVING],
                clear_devices=True,
                signature_def_map={'text_classification_tf_serving': signature}
            )

            # export model
            builder.save()
            print('Done exporting!')


if __name__ == '__main__':
    tf.app.run()
