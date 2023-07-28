import gevent.ssl
import numpy as np
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient
import tritonclient.grpc.aio as agrpcclient
from utils import add_system, build_prompt, str_as_bytes

from typing import Dict, AnyStr, List, Any

class HttpClient:
    def __init__(self,
        host="localhost",port=8501,
        ssl=False, key_file=None, cert_file=None, ca_certs=None, insecure=False,
        verbose=False):
        """

        :param url:
        :param ssl: Enable encrypted link to the server using HTTPS
        :param key_file: File holding client private key
        :param cert_file: File holding client certificate
        :param ca_certs: File holding ca certificate
        :param insecure: Use no peer verification in SSL communications. Use with caution
        :param verbose: Enable verbose output
        :return:
        """
        url = f"{host}:{port}"
        print("initializing-http from url=", url)

        if ssl:
            ssl_options = {}
            if key_file is not None:
                ssl_options['keyfile'] = key_file
            if cert_file is not None:
                ssl_options['certfile'] = cert_file
            if ca_certs is not None:
                ssl_options['ca_certs'] = ca_certs
            ssl_context_factory = None
            if insecure:
                ssl_context_factory = gevent.ssl._create_unverified_context
            triton_client = httpclient.InferenceServerClient(
                url=url,
                verbose=verbose,
                ssl=True,
                ssl_options=ssl_options,
                insecure=insecure,
                ssl_context_factory=ssl_context_factory)
        else:
            triton_client = httpclient.InferenceServerClient(
                url=url, verbose=verbose)

        self.triton_client = triton_client
    
    def infer(self, model_name,
          request_inputs:List[Dict[AnyStr, Any]], 
          request_outputs:List[AnyStr],
          request_compression_algorithm=None,
          response_compression_algorithm=None,
          streaming=False)-> httpclient.InferResult:
        """

        :param triton_client:
        :param model_name:
        :param request_compression_algorithm: Optional HTTP compression algorithm to use for the request body on client side.
                Currently supports "deflate", "gzip" and None. By default, no compression is used.
        :param response_compression_algorithm:
        :return:
        """
        
        inputs = []
        outputs = []
        # batch_size=4
        # 如果batch_size超过配置文件的max_batch_size，infer则会报错
        for item in request_inputs:
            name, data, dtype = item
            shape = data.shape
            inp =  httpclient.InferInput(
                name = name, shape= shape, datatype= dtype 
            )
            inp.set_data_from_numpy(data, binary_data=False)
            inputs.append(inp)
        
        for outname in request_outputs:
            out = httpclient.InferRequestedOutput(outname, binary_data=False)
            outputs.append(out)
        
        results = self.triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            request_compression_algorithm=request_compression_algorithm,
            response_compression_algorithm=response_compression_algorithm)
        # print(results)
        
        return results


class GrpcClient:
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
        triton_client = grpcclient.InferenceServerClient(
            url=url,
            verbose=verbose,
            ssl=ssl,
            root_certificates=root_certificates,
            private_key=private_key,
            certificate_chain=certificate_chain)

        self.triton_client = triton_client

    def infer(self, model_name,
          request_inputs:List[Dict[AnyStr, Any]], 
          request_outputs:List[AnyStr],
          compression_algorithm=None)->grpcclient.InferResult:
        """

        :param triton_client:
        :param model_name:
        :param request_compression_algorithm: Optional HTTP compression algorithm to use for the request body on client side.
                Currently supports "deflate", "gzip" and None. By default, no compression is used.
        :param response_compression_algorithm:
        :return:
        """
        
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
        
        
        results = self.triton_client.infer(
            model_name=model_name,
            inputs=inputs,
            outputs=outputs,
            compression_algorithm=compression_algorithm
        )
        # print(results)
        
        return results

            



def test_ensemble():
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
    
    client = HttpClient()
    # client = GrpcClient()

    results=  client.infer(
        model_name= "ensemblebloom",
        request_inputs=input_list,
        request_outputs=output_list
    )
    if isinstance(client , HttpClient):
        output = results.get_output(output_list[0])
        print(output)
        output = results.get_response()
        print(output)
    elif isinstance(client, GrpcClient):
        output = results.get_output(output_list[0], as_json=True)
        print(output)
        output = results.get_response(as_json=True)
        print(output)

    print("------")
    print(type(output))


def test_backend():
    import os
    from transformers import AutoTokenizer
    tokenizer_path = os.path.join(os.environ['HOME'], 'CommonModels',
            "xx-org/bloomS2.1" )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side= "right"
    input_ids = tokenizer([build_prompt(cnt) for cnt in content],
                           add_special_tokens=True,
                           padding=True,)
    print(input_ids)
    input_ids = input_ids.input_ids
    bsz= len(input_ids)
    print("bsz=", bsz, content)

    # input 0, 1, 2 is required
    input_list = [
        ("input_ids", np.array(input_ids).astype(np.uint32).reshape(bsz, -1), "UINT32" ) ,
        ("input_lengths", np.array(bsz*[len(input_ids)]).astype(np.uint32).reshape(bsz, -1), "UINT32"),
        ("request_output_len", np.array(bsz*[max_length]).astype(np.uint32).reshape(bsz, -1), "UINT32"),
        ("stop_words_list", np.array(bsz*[[],[]]).astype(np.int32).reshape(bsz,2, -1), "INT32" ) ,
        ("bad_words_list", np.array(bsz*[[], []]).astype(np.int32).reshape(bsz,2, -1), "INT32" ) ,

        ("start_id", np.array(bsz*[1]).astype(np.uint32).reshape(bsz, -1), "UINT32"),
        ("end_id", np.array(bsz*[2]).astype(np.uint32).reshape(bsz, -1), "UINT32"),
        ("runtime_top_p", np.array(bsz*[top_p]).astype(np.float32).reshape(bsz,-1), "FP32" ),
        ("temperature", np.array(bsz*[temperature]).astype(np.float32).reshape(bsz,-1), "FP32" ),
        ("random_seed", np.array(bsz*[seed ]).astype(np.uint64).reshape(bsz, -1), "UINT64")
    ]
    output_list = [
        "output_ids", 
        # "sequence_length",
        # "response_input_lengths", 
        # "cum_log_probs", "output_log_probs"
    ]
    
    client = HttpClient()
    # client = GrpcClient()

    results=  client.infer(
        model_name= "ensemblebloom",
        request_inputs=input_list,
        request_outputs=output_list
    )
    if isinstance(client , HttpClient):
        output = results.get_output(output_list[0])
        print(output)
        print(tokenizer.decode(output['data'], skip_special_tokens=True))
        output = results.get_response()
        print(output)
    elif isinstance(client, GrpcClient):
        output = results.get_output(output_list[0], as_json=True)
        print(output)
        output = results.get_response(as_json=True)
        print(output)

    print("------")
    print(type(output))



if __name__ == "__main__":
    import random
    import numpy as np
    content = ["用200字介绍一下北京" , "hi，你好!"]
    max_length = 128
    top_p = 0.6
    temperature = 0.95
    seed= random.randint(1, 100000)
    
    # test_backend()
    test_ensemble()
