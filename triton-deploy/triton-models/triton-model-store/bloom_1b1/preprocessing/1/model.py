# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
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

import csv
import json

import numpy as np
import torch
import triton_python_backend_utils as pb_utils
from transformers import AutoTokenizer


def to_word_list_format(tokenizer, word_dict):
    flat_ids = []
    offsets = []
    for word_dict_item in word_dict:
        item_flat_ids = []
        item_offsets = []

        if isinstance(word_dict_item[0], bytes):
            word_dict_item = [word_dict_item[0].decode()]

        words = list(csv.reader(word_dict_item))[0]
        for word in words:
            ids = tokenizer.encode(word)

            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))


class TritonPythonModel:
    """Your Python model must use the same class name. Every Python model
    that is created must have "TritonPythonModel" as the class name.
    """

    def initialize(self, args):
        """`initialize` is called only once when the model is being loaded.
        Implementing `initialize` function is optional. This function allows
        the model to initialize any state associated with this model.
        Parameters
        ----------
        args : dict
          Both keys and values are strings. The dictionary keys and values are:
          * model_config: A JSON string containing the model configuration
          * model_instance_kind: A string containing model instance kind
          * model_instance_device_id: A string containing model instance device ID
          * model_repository: Model repository path
          * model_version: Model version
          * model_name: Model name
        """
        # Parse model configs
        self.model_config = model_config = json.loads(args['model_config'])

        # Parse model output configs and convert Triton types to numpy types
        input_names = ["INPUT_ID", "REQUEST_INPUT_LEN", "BAD_WORDS_IDS", "STOP_WORDS_IDS"]
        for input_name in input_names:
          setattr(self,
              input_name.lower() + "_dtype",
              pb_utils.triton_string_to_numpy(pb_utils.get_output_config_by_name(
                model_config, input_name)['data_type'])
          )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["parameters"]["tokenizer_path"]["string_value"], padding_side='right')
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def execute(self, requests):
        """`execute` must be implemented in every Python model. `execute`
        function receives a list of pb_utils.InferenceRequest as the only
        argument. This function is called when an inference is requested
        for this model. Depending on the batching configuration (e.g. Dynamic
        Batching) used, `requests` may contain multiple requests. Every
        Python model, must create one pb_utils.InferenceResponse for every
        pb_utils.InferenceRequest in `requests`. If there is an error, you can
        set the error argument when creating a pb_utils.InferenceResponse.
        Parameters
        ----------
        requests : list
          A list of pb_utils.InferenceRequest
        Returns
        -------
        list
          A list of pb_utils.InferenceResponse. The length of this list must
          be the same as `requests`
        """

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for idx, request in enumerate(requests):
            # Get input tensors
            query = pb_utils.get_input_tensor_by_name(request, 'QUERY').as_numpy()
            request_output_len = pb_utils.get_input_tensor_by_name(request, 'REQUEST_OUTPUT_LEN').as_numpy()

            bad_words_dict = pb_utils.get_input_tensor_by_name(request, 'BAD_WORDS_DICT').as_numpy()
            stop_words_dict = pb_utils.get_input_tensor_by_name(request, 'STOP_WORDS_DICT').as_numpy()

            # Preprocessing input data.
            input_id, request_input_len = self._create_request(query)
            bad_words = to_word_list_format(self.tokenizer, bad_words_dict)
            stop_words = to_word_list_format(self.tokenizer, stop_words_dict)

            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            input_id_tensor = pb_utils.Tensor(
                'INPUT_ID',
                np.array(input_id).astype(self.input_id_dtype))
            request_input_len_tensor = pb_utils.Tensor(
                'REQUEST_INPUT_LEN',
                np.array(request_input_len).astype(self.request_input_len_dtype))
            request_output_len_tensor = pb_utils.Tensor(
                'REQUEST_OUTPUT_LEN',
                request_output_len)
            bad_words_ids_tensor = pb_utils.Tensor(
                'BAD_WORDS_IDS',
                bad_words)
            stop_words_ids_tensor = pb_utils.Tensor(
                'STOP_WORDS_IDS',
                stop_words)

            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(output_tensors=[
                input_id_tensor,
                bad_words_ids_tensor,
                stop_words_ids_tensor,
                request_input_len_tensor,
                request_output_len_tensor])
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')

    def _create_request(self, query):
        """
        query : batch string (2D numpy array)
        """

        # 2D numpy string bytes to 1D list of strings.
        inputs  = [s[0].decode() for s in query.tolist()]
        encoded_inputs = self.tokenizer(inputs, padding=True, return_tensors='pt')
        input_token_ids = encoded_inputs['input_ids'].int()
        input_lengths = encoded_inputs['attention_mask'].sum(
            dim=-1, dtype=torch.int32).view(-1, 1)
        return input_token_ids, input_lengths

    def _create_word_list(self, word_dict):
        flat_ids = []
        offsets = []
        for word_dict_item in word_dict:
            item_flat_ids = []
            item_offsets = []

            words = list(csv.reader([word_dict_item[0].decode()]))[0]
            for word in words:
                ids = self._encode(word)

                if len(ids) == 0:
                    continue

                item_flat_ids += ids
                item_offsets.append(len(ids))

            flat_ids.append(np.array(item_flat_ids))
            offsets.append(np.cumsum(np.array(item_offsets)))

        pad_to = max(1, max(len(ids) for ids in flat_ids))

        for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
            flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
            offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

        return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))

    def _encode(self, sentence):
        sentence = sentence.decode() if isinstance(sentence, bytes) else sentence
        return self.tokenizer.encode(sentence)
