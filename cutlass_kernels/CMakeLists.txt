# Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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


include_directories(../cutlass_extensions/include/)
include_directories(${CUTLASS_DIR}/include)
include_directories(../utils)
include_directories(../)
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/include)
link_directories(${CUDA_TOOLKIT_ROOT_DIR}/lib64)

find_package(Python REQUIRED)
include_directories(~/miniconda3/envs/tgi/include/python3.10/)
# include_directories(${PYTHON_INCLUDE_DIRS})

execute_process(
    COMMAND
        ${Python_EXECUTABLE} -c
            "import torch.utils; print(torch.utils.cmake_prefix_path)"
    OUTPUT_STRIP_TRAILING_WHITESPACE
    OUTPUT_VARIABLE DCMAKE_PREFIX_PATH)
set(CMAKE_PREFIX_PATH "${DCMAKE_PREFIX_PATH}")

find_package(Torch REQUIRED)
# include_directories(~/miniconda3/envs/tgi/lib/python3.10/site-packages/torch/include/)
include_directories(${TORCH_INCLUDE_DIRS})
find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_INSTALL_PREFIX}/lib")

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -t 8 \
                      -gencode=arch=compute_70,code=\\\"sm_70,compute_70\\\" \
                      -gencode=arch=compute_75,code=\\\"sm_75,compute_75\\\" \
                      -gencode=arch=compute_80,code=\\\"sm_80,compute_80\\\" \
                      ")

# add_library(fpA_intB_gemm SHARED fused_gemm_dq.cc fpA_intB_gemm.cu cutlass_heuristic.cc cutlass_preprocessors.cc ${CMAKE_CURRENT_SOURCE_DIR}/../utils/logger.cc ${CMAKE_CURRENT_SOURCE_DIR}/../utils/cuda_utils.cc)
add_library(fpA_intB_gemm SHARED fused_gemm_dq.cc cutlass_heuristic.cc cutlass_preprocessors.cc ${CMAKE_CURRENT_SOURCE_DIR}/../utils/logger.cc ${CMAKE_CURRENT_SOURCE_DIR}/../utils/cuda_utils.cc)
target_link_libraries(fpA_intB_gemm "${TORCH_LIBRARIES}" "${TORCH_PYTHON_LIBRARY}")
