https_proxy="socks5h://127.0.0.1:1080"

# FasterTransformer
git clone https://github.com/NVIDIA/FasterTransformer.git
cd FasterTransformer/build
git submodule init && git submodule update
cmake -DSM=70 -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -D CMAKE_EXPORT_COMPILE_COMMANDS=1 -D CMAKE_INSTALL_PREFIX=/opt/tritonserver -D TRITON_COMMON_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" -D TRITON_CORE_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}"  -D TRITON_BACKEND_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}"  ..
make -j32 install

# fastertransformer_backend
git clone https://github.com/triton-inference-server/fastertransformer_backend.git
cmake \
      -D CMAKE_EXPORT_COMPILE_COMMANDS=1 \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=/opt/tritonserver \
      -D TRITON_COMMON_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      -D TRITON_CORE_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      -D TRITON_BACKEND_REPO_TAG="r${NVIDIA_TRITON_SERVER_VERSION}" \
      .. && \
    make -j"$(grep -c ^processor /proc/cpuinfo)" install


CUDA_VISIBLE_DEVICES=0 /opt/tritonserver/bin/tritonserver \
    --model-repository=/export/triton-model-store/bloom_1b1/ \
    --backend-config=python,shm-region-prefix-name=prefix1_ \
    --grpc-port 8500 --http-port 8501 --metrics-port 12345 \
    --log-verbose 1 --log-file /export/Logs/triton_server_gpu0.log
