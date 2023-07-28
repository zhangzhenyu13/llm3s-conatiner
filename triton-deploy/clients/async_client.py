import queue
from functools import partial
import gevent.ssl
import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
from tritonclient import grpc
import tritonclient.grpc.aio as agrpcclient
from tritonclient.utils import InferenceServerException

from utils import add_system, build_prompt, str_as_bytes

from typing import Dict, AnyStr, List, Any

class UserData:
    def __init__(self):
        self._completed_requests = queue.Queue()
def callback(user_data, result:grpcclient.InferResult, error):
    if error:
        print("e:",error)
        user_data._completed_requests.put(error)
    else:
        modelres: grpc.service_pb2.ModelInferResponse = result.get_response()
        print("r:",result, type(result.get_response(as_json=False)))
        user_data._completed_requests.put(result)
class AysncGrpcClient:
    def __init__(self,
        host="localhost",port=8500,
        ssl=False, private_key=None, root_certificates=None, certificate_chain=None,
        verbose=False):
        """

        :param url:
        :param ssl: Enable SSL encrypted channel to the server
        :param private_key: File holding PEM-encoded private key
        :param root_certificates: File holding PEM-encoded root certificates
        :param certificate_chain: File holding PEM-encoded certicate chain
        :param verbose:
        :return:
        """
        url = f"{host}:{port}"
        print("initializing-grpc from url=", url)
        self.params_ = {
            "url":url,
            "verbose":verbose,
            # "ssl":ssl,
            # "root_certificates":root_certificates,
            # "private_key":private_key,
            # "certificate_chain":certificate_chain
        }
    async def stream_infer_v1(self, model_name,
          request_inputs:List[Dict[AnyStr, Any]], 
          request_outputs:List[AnyStr],
          sequence_id=None,
          compression_algorithm=None)->grpcclient.InferResult:
        
        inputs = []
        outputs = []
        # batch_size=4
        # 如果batch_size超过配置文件的max_batch_size，infer则会报错
        for item in request_inputs:
            name, data, dtype = item
            shape = data.shape
            inp =  grpcclient.InferInput(
                name = name, shape= shape, datatype= dtype 
            )
            inp.set_data_from_numpy(data)
            inputs.append(inp)
        
        for outname in request_outputs:
            out = grpcclient.InferRequestedOutput(outname)
            outputs.append(out)
        user_data = UserData()
        try:
            with grpcclient.InferenceServerClient(**self.params_) as triton_client:
                triton_client.start_stream(callback=partial(callback, user_data))

                triton_client.async_stream_infer(
                    model_name=model_name,
                    inputs=inputs,
                    outputs=outputs,
                    # request_id="{}_{}".format(sequence_id, count),
                    # sequence_id=sequence_id,
                    # sequence_start=(count == 1),
                    # sequence_end=(count == len(values)),
                )
        except InferenceServerException as error:
            print(error)
            raise error
        
        # Retrieve results...
        recv_count = 0
        while True:
            print("recv:", recv_count)
            data_item = user_data._completed_requests.get()
            if type(data_item) == InferenceServerException:
                print(data_item)
                
            else:
                try:
                    result = data_item.get_output(outputs[0])
                    print(result)
                except ValueError as e:
                    raise e
            recv_count = recv_count + 1



    async def stream_infer(self, model_name,
          request_inputs:List[Dict[AnyStr, Any]], 
          request_outputs:List[AnyStr],
          compression_algorithm=None)->grpcclient.InferResult:
        
        inputs = []
        outputs = []
        # batch_size=4
        # 如果batch_size超过配置文件的max_batch_size，infer则会报错
        for item in request_inputs:
            name, data, dtype = item
            shape = data.shape
            inp =  grpcclient.InferInput(
                name = name, shape= shape, datatype= dtype 
            )
            inp.set_data_from_numpy(data)
            inputs.append(inp)
        
        for outname in request_outputs:
            out = grpcclient.InferRequestedOutput(outname)
            outputs.append(out)
        
        async with agrpcclient.InferenceServerClient(**self.params_) as triton_client:
            async def request_itereator():
                request_data={
                    "model_name": model_name,
                    "inputs": inputs,
                    "outputs": outputs
                }
                yield request_data
            try:
                print("1.create client")
                response_iterator = triton_client.stream_infer(
                    inputs_iterator=request_itereator(), stream_timeout=None
                )
                print("wait for response")
                async for response in response_iterator:
                    result, error = response
                    if error:
                        print("error=", error)
                    else:
                        for outname in outputs:
                            print(result.get_output(outname) )
            except Exception as e:
                print("except:",e.args)
                raise e
            



async def test_ensemble():
    # input 0, 1, 2 is required
    bsz = len(content)
    input0 = [str_as_bytes(build_prompt(cnt)) for cnt in content]
    input_list = [
        ("INPUT_0", np.array(input0).astype(np.bytes_).reshape(bsz, -1), "BYTES" ) ,
        ("INPUT_1", np.array(bsz* [max_length]).astype(np.uint32).reshape(bsz, -1), "UINT32"),
        ("INPUT_2", np.array(bsz*[str_as_bytes("")]).astype(np.bytes_).reshape(bsz, -1), "BYTES" ) ,
        ("INPUT_3", np.array(bsz*[str_as_bytes("")]).astype(np.bytes_).reshape(bsz, -1), "BYTES" ) ,
        ("start_id", np.array(bsz*[1]).astype(np.uint32).reshape(bsz, -1), "UINT32"),
        ("end_id", np.array(bsz*[2]).astype(np.uint32).reshape(bsz, -1), "UINT32"),
        ("runtime_top_p", np.array(bsz*[top_p]).astype(np.float32).reshape(bsz,-1), "FP32" ),
        ("temperature", np.array(bsz*[temperature]).astype(np.float32).reshape(bsz,-1), "FP32" ),
        ("random_seed", np.array(bsz*[seed ]).astype(np.uint64).reshape(bsz, -1), "UINT64")
    ]
    output_list = [
        "OUTPUT_0"
    ]
    
    client = AysncGrpcClient()
    

    results=  await client.stream_infer_v1(
        model_name= "ensemblebloom",
        request_inputs=input_list,
        request_outputs=output_list
    )

    output = results.get_output(output_list[0], as_json=True)
    print(output)
    output = results.get_response(as_json=True)
    print(output)

    print("------")
    print(type(output))

if __name__ == "__main__":
    import random
    import numpy as np
    import asyncio
    content = ["用200字介绍一下北京" ] #, "hi，你好!"]
    max_length = 128
    top_p = 0.6
    temperature = 0.95
    seed= random.randint(1, 100000)

    asyncio.run(test_ensemble())
