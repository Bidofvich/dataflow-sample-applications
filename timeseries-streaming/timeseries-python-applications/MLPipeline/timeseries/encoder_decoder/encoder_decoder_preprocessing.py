# Lint as: python2, python3
# Copyright 2020 Google LLC. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import math
from datetime import datetime
from typing import Dict, Text, Any, Optional

import tensorflow as tf
import tensorflow_transform as tft
from timeseries.utils import timeseries_transform_utils


def preprocessing_fn(inputs: Dict[Text, Any], custom_config: Dict[Text, Any]) -> Dict[Text, Any]:
    """tf.transform's callback function for preprocessing inputs.

    Args:
      inputs: map from feature keys to raw not-yet-transformed features.
      custom_config:
        timesteps: The number of timesteps in the look back window
        features: Which of the features from the TF.Example to use in the model.

    Returns:
      Map from string feature key to transformed feature operations.
    """
    timesteps = custom_config['timesteps']

    outputs = inputs.copy()
    for key in outputs:
        outputs[key] = tf.sparse.to_dense(outputs[key])

    """
    Scale the inputs with the exception of TIMESTAMPS
    """
    for key in outputs:
        if not str(key).endswith('_TIMESTAMP'):
            outputs[key] = tft.scale_to_z_score(outputs[key])

    # Generate features to be used in the model
    train_x_tensors = timeseries_transform_utils.create_feature_list_from_dict(outputs, custom_config)

    train_x_values = [train_x_tensors[k] for k in sorted(train_x_tensors)]

    float32 = tf.reshape(
        tf.stack(train_x_values, axis=-1),
        [-1, timesteps, len(train_x_values)])

    # AutoEncoder / AutoDecoder requires label == data
    outputs = {'Float32': float32, 'LABEL': float32}
    return outputs
