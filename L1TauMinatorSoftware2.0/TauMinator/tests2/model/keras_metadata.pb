
�xroot"_tf_keras_network*�x{"name": "TauMinator_CB_calib", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "TauMinator_CB_calib", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "TowerClusterImage"}, "name": "TowerClusterImage", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "TowerClusterPosition"}, "name": "TowerClusterPosition", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "TowerClusterEnergy"}, "name": "TowerClusterEnergy", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "middleMan", "trainable": true, "dtype": "float32", "axis": 1}, "name": "middleMan", "inbound_nodes": [[["TowerClusterImage", 0, 0, {}], ["TowerClusterPosition", 0, 0, {}], ["TowerClusterEnergy", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "CALlayer1", "trainable": true, "dtype": "float32", "units": 180, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CALlayer1", "inbound_nodes": [[["middleMan", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "RELU_CALlayer1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "RELU_CALlayer1", "inbound_nodes": [[["CALlayer1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "CALlayer2", "trainable": true, "dtype": "float32", "units": 75, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CALlayer2", "inbound_nodes": [[["RELU_CALlayer1", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "RELU_CALlayer2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "RELU_CALlayer2", "inbound_nodes": [[["CALlayer2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "CALlayer3", "trainable": true, "dtype": "float32", "units": 75, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CALlayer3", "inbound_nodes": [[["RELU_CALlayer2", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "RELU_CALlayer3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "RELU_CALlayer3", "inbound_nodes": [[["CALlayer3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "CALlayer3p", "trainable": true, "dtype": "float32", "units": 75, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CALlayer3p", "inbound_nodes": [[["RELU_CALlayer3", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "RELU_CALlayer3p", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "RELU_CALlayer3p", "inbound_nodes": [[["CALlayer3p", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "CALlayer3s", "trainable": true, "dtype": "float32", "units": 75, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CALlayer3s", "inbound_nodes": [[["RELU_CALlayer3p", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "RELU_CALlayer3s", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "RELU_CALlayer3s", "inbound_nodes": [[["CALlayer3s", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "CALlayer3t", "trainable": true, "dtype": "float32", "units": 75, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CALlayer3t", "inbound_nodes": [[["RELU_CALlayer3s", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "RELU_CALlayer3t", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "RELU_CALlayer3t", "inbound_nodes": [[["CALlayer3t", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "CALout", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CALout", "inbound_nodes": [[["RELU_CALlayer3t", 0, 0, {}]]]}, {"class_name": "Activation", "config": {"name": "LIN_CALlayer3", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "LIN_CALlayer3", "inbound_nodes": [[["CALout", 0, 0, {}]]]}], "input_layers": [["TowerClusterImage", 0, 0], ["TowerClusterPosition", 0, 0], ["TowerClusterEnergy", 0, 0]], "output_layers": [["LIN_CALlayer3", 0, 0]]}, "shared_object_id": 32, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 5]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 2]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}, {"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 1]}, "ndim": 2, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": [{"class_name": "TensorShape", "items": [null, 5]}, {"class_name": "TensorShape", "items": [null, 2]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "full_save_spec": {"class_name": "__tuple__", "items": [[[{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 5]}, "float32", "TowerClusterImage"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 2]}, "float32", "TowerClusterPosition"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "TowerClusterEnergy"]}]], {}]}, "save_spec": [{"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 5]}, "float32", "TowerClusterImage"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 2]}, "float32", "TowerClusterPosition"]}, {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 1]}, "float32", "TowerClusterEnergy"]}], "keras_version": "2.6.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "TauMinator_CB_calib", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "TowerClusterImage"}, "name": "TowerClusterImage", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "TowerClusterPosition"}, "name": "TowerClusterPosition", "inbound_nodes": [], "shared_object_id": 1}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "TowerClusterEnergy"}, "name": "TowerClusterEnergy", "inbound_nodes": [], "shared_object_id": 2}, {"class_name": "Concatenate", "config": {"name": "middleMan", "trainable": true, "dtype": "float32", "axis": 1}, "name": "middleMan", "inbound_nodes": [[["TowerClusterImage", 0, 0, {}], ["TowerClusterPosition", 0, 0, {}], ["TowerClusterEnergy", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "Dense", "config": {"name": "CALlayer1", "trainable": true, "dtype": "float32", "units": 180, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CALlayer1", "inbound_nodes": [[["middleMan", 0, 0, {}]]], "shared_object_id": 6}, {"class_name": "Activation", "config": {"name": "RELU_CALlayer1", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "RELU_CALlayer1", "inbound_nodes": [[["CALlayer1", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "Dense", "config": {"name": "CALlayer2", "trainable": true, "dtype": "float32", "units": 75, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CALlayer2", "inbound_nodes": [[["RELU_CALlayer1", 0, 0, {}]]], "shared_object_id": 10}, {"class_name": "Activation", "config": {"name": "RELU_CALlayer2", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "RELU_CALlayer2", "inbound_nodes": [[["CALlayer2", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Dense", "config": {"name": "CALlayer3", "trainable": true, "dtype": "float32", "units": 75, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CALlayer3", "inbound_nodes": [[["RELU_CALlayer2", 0, 0, {}]]], "shared_object_id": 14}, {"class_name": "Activation", "config": {"name": "RELU_CALlayer3", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "RELU_CALlayer3", "inbound_nodes": [[["CALlayer3", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Dense", "config": {"name": "CALlayer3p", "trainable": true, "dtype": "float32", "units": 75, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CALlayer3p", "inbound_nodes": [[["RELU_CALlayer3", 0, 0, {}]]], "shared_object_id": 18}, {"class_name": "Activation", "config": {"name": "RELU_CALlayer3p", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "RELU_CALlayer3p", "inbound_nodes": [[["CALlayer3p", 0, 0, {}]]], "shared_object_id": 19}, {"class_name": "Dense", "config": {"name": "CALlayer3s", "trainable": true, "dtype": "float32", "units": 75, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}, "shared_object_id": 20}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 21}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CALlayer3s", "inbound_nodes": [[["RELU_CALlayer3p", 0, 0, {}]]], "shared_object_id": 22}, {"class_name": "Activation", "config": {"name": "RELU_CALlayer3s", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "RELU_CALlayer3s", "inbound_nodes": [[["CALlayer3s", 0, 0, {}]]], "shared_object_id": 23}, {"class_name": "Dense", "config": {"name": "CALlayer3t", "trainable": true, "dtype": "float32", "units": 75, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CALlayer3t", "inbound_nodes": [[["RELU_CALlayer3s", 0, 0, {}]]], "shared_object_id": 26}, {"class_name": "Activation", "config": {"name": "RELU_CALlayer3t", "trainable": true, "dtype": "float32", "activation": "relu"}, "name": "RELU_CALlayer3t", "inbound_nodes": [[["CALlayer3t", 0, 0, {}]]], "shared_object_id": 27}, {"class_name": "Dense", "config": {"name": "CALout", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "CALout", "inbound_nodes": [[["RELU_CALlayer3t", 0, 0, {}]]], "shared_object_id": 30}, {"class_name": "Activation", "config": {"name": "LIN_CALlayer3", "trainable": true, "dtype": "float32", "activation": "sigmoid"}, "name": "LIN_CALlayer3", "inbound_nodes": [[["CALout", 0, 0, {}]]], "shared_object_id": 31}], "input_layers": [["TowerClusterImage", 0, 0], ["TowerClusterPosition", 0, 0], ["TowerClusterEnergy", 0, 0]], "output_layers": [["LIN_CALlayer3", 0, 0]]}}}2
�root.layer-0"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "TowerClusterImage", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 5]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "TowerClusterImage"}}2
�root.layer-1"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "TowerClusterPosition", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "TowerClusterPosition"}}2
�root.layer-2"_tf_keras_input_layer*�{"class_name": "InputLayer", "name": "TowerClusterEnergy", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "TowerClusterEnergy"}}2
�root.layer-3"_tf_keras_layer*�{"name": "middleMan", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Concatenate", "config": {"name": "middleMan", "trainable": true, "dtype": "float32", "axis": 1}, "inbound_nodes": [[["TowerClusterImage", 0, 0, {}], ["TowerClusterPosition", 0, 0, {}], ["TowerClusterEnergy", 0, 0, {}]]], "shared_object_id": 3, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 5]}, {"class_name": "TensorShape", "items": [null, 2]}, {"class_name": "TensorShape", "items": [null, 1]}]}2
�root.layer_with_weights-0"_tf_keras_layer*�{"name": "CALlayer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "CALlayer1", "trainable": true, "dtype": "float32", "units": 180, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["middleMan", 0, 0, {}]]], "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 8}}, "shared_object_id": 36}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8]}}2
�root.layer-5"_tf_keras_layer*�{"name": "RELU_CALlayer1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "RELU_CALlayer1", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["CALlayer1", 0, 0, {}]]], "shared_object_id": 7}2
�root.layer_with_weights-1"_tf_keras_layer*�{"name": "CALlayer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "CALlayer2", "trainable": true, "dtype": "float32", "units": 75, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["RELU_CALlayer1", 0, 0, {}]]], "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 180}}, "shared_object_id": 37}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 180]}}2
�root.layer-7"_tf_keras_layer*�{"name": "RELU_CALlayer2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "RELU_CALlayer2", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["CALlayer2", 0, 0, {}]]], "shared_object_id": 11}2
�	root.layer_with_weights-2"_tf_keras_layer*�{"name": "CALlayer3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "CALlayer3", "trainable": true, "dtype": "float32", "units": 75, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}, "shared_object_id": 12}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 13}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["RELU_CALlayer2", 0, 0, {}]]], "shared_object_id": 14, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 75}}, "shared_object_id": 38}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75]}}2
�
root.layer-9"_tf_keras_layer*�{"name": "RELU_CALlayer3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Activation", "config": {"name": "RELU_CALlayer3", "trainable": true, "dtype": "float32", "activation": "relu"}, "inbound_nodes": [[["CALlayer3", 0, 0, {}]]], "shared_object_id": 15}2
�root.layer_with_weights-3"_tf_keras_layer*�{"name": "CALlayer3p", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "CALlayer3p", "trainable": true, "dtype": "float32", "units": 75, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["RELU_CALlayer3", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 75}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75]}}2
�
�
�
�root.layer_with_weights-5"_tf_keras_layer*�{"name": "CALlayer3t", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "CALlayer3t", "trainable": true, "dtype": "float32", "units": 75, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}, "shared_object_id": 24}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 25}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["RELU_CALlayer3s", 0, 0, {}]]], "shared_object_id": 26, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 75}}, "shared_object_id": 41}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75]}}2
�
�root.layer_with_weights-6"_tf_keras_layer*�{"name": "CALout", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "CALout", "trainable": true, "dtype": "float32", "units": 1, "activation": "linear", "use_bias": false, "kernel_initializer": {"class_name": "RandomNormal", "config": {"mean": 0.0, "stddev": 0.05, "seed": 7}, "shared_object_id": 28}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 29}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["RELU_CALlayer3t", 0, 0, {}]]], "shared_object_id": 30, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 75}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 75]}}2
�
��root.keras_api.metrics.0"_tf_keras_metric*�{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 43}2
��root.keras_api.metrics.1"_tf_keras_metric*�{"class_name": "RootMeanSquaredError", "name": "root_mean_squared_error", "dtype": "float32", "config": {"name": "root_mean_squared_error", "dtype": "float32"}, "shared_object_id": 44}2