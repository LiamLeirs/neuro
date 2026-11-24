FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    ninja-build \
    git \
    wget \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN wget -nv -O /tmp/arrayfire.sh "https://arrayfire.gateway.scarf.sh/linux/3.10.0/ArrayFire.sh" \
    && bash /tmp/arrayfire.sh \
        --skip-license \
        --prefix=/opt \
        --include-subdir \
    && rm /tmp/arrayfire.sh

ENV CMAKE_PREFIX_PATH=/opt/ArrayFire-3.10.0-Linux

RUN ln -s libcuda.so \
    /usr/local/cuda/targets/x86_64-linux/lib/stubs/libcuda.so.1

WORKDIR /workspace

COPY . .

RUN cmake -S . -B build -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DNEURO_BUILD_EXAMPLES=ON \
    -DNEURO_BUILD_TESTS=ON \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/targets/x86_64-linux/lib/stubs"

RUN cmake --build build --parallel 2