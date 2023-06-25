#include <cublas_v2.h>
#include <iostream>
#include <vector>

#include "torch/csrc/cuda/Stream.h"
#include <torch/custom_class.h>
#include <torch/script.h>

#include "cutlass_kernels/fpA_intB_gemm/fpA_intB_gemm.h"
#include "cutlass_kernels/th_utils.h"

#include "cutlass/numeric_types.h"

using torch::Tensor;

namespace fastertransformer {

template<typename T, typename WeightType>
Tensor fused_gemm_dq_helper(
    Tensor input_activations, Tensor weight, Tensor scales)
{
    const at::ScalarType _st    = input_activations.scalar_type();
    const int            m      = input_activations.size(0);
    const int            n      = scales.size(0);
    const int            k      = input_activations.size(1);
    auto                 stream = at::cuda::getCurrentCUDAStream().stream();

    const T*          input_act_ptr = get_ptr<const T>(input_activations);
    const WeightType* weight_ptr    = get_ptr<const WeightType>(weight);
    const T*          scales_ptr    = get_ptr<const T>(scales);

    CutlassFpAIntBGemmRunner<T, WeightType> fused_gemm_dq_runner;
    const int ws_bytes = fused_gemm_dq_runner.getWorkspaceSize(m, n, k);

    auto output_tensor = torch::empty({m, n}, torch::dtype(_st).device(torch::kCUDA).requires_grad(false));
    auto ws_tensor     = torch::empty({ws_bytes}, torch::dtype(torch::kInt8).device(torch::kCUDA).requires_grad(false));

    T*   output_tensor_ptr = get_ptr<T>(output_tensor);
    char* ws_ptr            = get_ptr<char>(ws_tensor);

    fused_gemm_dq_runner.gemm(input_act_ptr, weight_ptr, scales_ptr, output_tensor_ptr, m, n, k, ws_ptr, ws_bytes, stream);
    return output_tensor;
}

Tensor fused_gemm_dq(Tensor input_activations, Tensor weight, Tensor scales)
{
    const at::ScalarType _st = input_activations.scalar_type();
    CHECK_INPUT(scales, _st);

    TORCH_CHECK(input_activations.dim() == 2, "Invalid rank for activations");
    TORCH_CHECK(weight.dim() == 2, "Invalid rank for weight");
    TORCH_CHECK(scales.dim() == 1, "Invalid rank for scales");

    const int m = input_activations.size(0);
    const int n = scales.size(0);
    const int k = input_activations.size(1);

    TORCH_CHECK(input_activations.size(1) == weight.size(0), "dim 1 of act and dim 0 of weight must be equal");

    // We signal int4 by having the last weight dim be half the size of the scales.
    // This is because int4 elements are packed into a single byte.
    torch::ScalarType quant_type = weight.scalar_type();
    if (weight.size(-1) == scales.size(-1) / 2) {
        quant_type = at::ScalarType::QUInt4x2;
    }
    else {
        TORCH_CHECK(weight.size(-1) == scales.size(-1),
                    "Last dim of weight and scales must be equal for int8 "
                    "or last dim of scale must be 2x last dim of weight for int4.");
    }

    Tensor output_tensor;
    switch (_st) {
        case at::ScalarType::Half: {
            if (quant_type == torch::kInt8) {
                output_tensor =
                    fused_gemm_dq_helper<half, uint8_t>(input_activations, weight, scales);
            }
            else if (quant_type == at::ScalarType::QUInt4x2) {
                output_tensor = fused_gemm_dq_helper<half, cutlass::uint4b_t>(
                    input_activations, weight, scales);
            }
            else {
                std::string err_msg = "Unsupported weight type " + std::string(at::toString(quant_type));
                throw std::runtime_error(err_msg);
            }
            break;
        }
        default:
            throw std::runtime_error("Unsupported tensor type. Got " + std::string(at::toString(_st)));
    }
    return output_tensor;
}
TORCH_LIBRARY(gemm_dq_unit_ops, m)
{
    m.def("fused_gemm_dq", fused_gemm_dq);
}
}
