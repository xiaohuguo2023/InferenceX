// SPDX-License-Identifier: Apache-2.0
// Native HIP port of vLLM's csrc/libtorch_stable/attention/dcp_utils/
// dcp_direct_a2a_lse_reduce.cu  (hand-rewritten for ROCm/MI355X gfx950).
//
// Registers torch.ops._C.direct_dcp_a2a_lse_reduce for the CUDA (== HIP) key so
// vLLM's MLADCPManager direct-combine path runs on ROCm with NO host sync:
// all 4 kernels are enqueued on the current stream (naturally ordered after the
// MLA forward_mqa producer + fold un-fold copy), and cross-rank ordering is done
// on-GPU via the epoch/system-scope-signal handshake -- the exact mechanism NV
// uses. Load with torch.ops.load_library(".../dcp_direct_a2a_lse_reduce.so").
//
// Kernel bodies are a verbatim port of the CUDA original; only the includes,
// the release/acquire primitives (dcp_direct_common_hip.h), the torch tensor API
// (stable -> regular at::Tensor), and the stream/guard/launch-check are HIP.
#include <hip/hip_bf16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPStream.h>
#include <torch/library.h>

#include <cstdio>
#include <cstdlib>
#include <optional>
#include <type_traits>

#include "dcp_direct_common_hip.h"

namespace {

using vllm::direct_dcp::check_hip_launch;
using vllm::direct_dcp::get_peer_ptr;
using vllm::direct_dcp::increment_epoch_kernel;
using vllm::direct_dcp::store_release_system;
using vllm::direct_dcp::wait_for_epoch;

__device__ __forceinline__ float k_pos_inf() {
  return __int_as_float(0x7f800000);
}
// log2(e), used to convert natural-base LSE to base-2 for stable exp2 combine.
constexpr float K_LOG2E_F = 1.4426950408889634f;

__device__ __forceinline__ int64_t find_sequence(const int32_t* query_start_loc,
                                                 int64_t token_idx,
                                                 int64_t num_seqs) {
  int64_t left = 0;
  int64_t right = num_seqs;
  while (left < right) {
    int64_t mid = (left + right) / 2;
    if (query_start_loc[mid] <= token_idx) {
      left = mid + 1;
    } else {
      right = mid;
    }
  }
  return left - 1;
}

template <typename scalar_t>
__device__ __forceinline__ float to_float(scalar_t value) {
  if constexpr (std::is_same_v<scalar_t, float>) {
    return value;
  } else if constexpr (std::is_same_v<scalar_t, __half>) {
    return __half2float(value);
  } else {
    return __bfloat162float(value);
  }
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t from_float(float value) {
  if constexpr (std::is_same_v<scalar_t, __half>) {
    return __float2half_rn(value);
  } else {
    return __float2bfloat16(value);
  }
}

// Send each destination's head slice and LSE to its symmetric buffer. Mask
// undefined empty-shard rows at the source so peers give them zero weight.
template <typename lse_t>
__global__ void dispatch_output_lse_kernel(
    const uint4* partial_output, const lse_t* partial_lse,
    const int32_t* seq_lens, const int32_t* query_start_loc,
    const int64_t* peer_output_ptrs, const int64_t* peer_lse_ptrs,
    const int64_t* epoch_ptr, int64_t world_size, int64_t rank,
    int64_t num_tokens, int64_t max_num_tokens, int64_t num_seqs,
    int64_t heads_per_rank, int64_t head_dim, int64_t output_token_stride,
    int64_t lse_token_stride, bool report_out_of_range) {
  int64_t item = static_cast<int64_t>(blockIdx.x);
  int64_t destination_rank = item / num_tokens;
  int64_t token_idx = item - destination_rank * num_tokens;

  __shared__ bool empty_kv;
  if (threadIdx.x == 0) {
    empty_kv = false;
    if (seq_lens != nullptr) {
      int64_t seq_idx = find_sequence(query_start_loc, token_idx, num_seqs);
      // find_sequence returns `left - 1`, so it yields -1 for any token that
      // sorts before query_start_loc[0]. Every other index in this op is
      // pinned by the host-side TORCH_CHECKs, which makes seq_lens[seq_idx]
      // the ONLY load here that can leave its allocation -- and a load 4 bytes
      // below a page-aligned tensor faults outright rather than reading junk.
      // The condition is reachable: the DSpark draft enters this op with its
      // own metadata, and during warmup/cudagraph capture those persistent
      // buffers can still hold stale boundaries that do not start at 0.
      if (seq_idx < 0 || seq_idx >= num_seqs) {
        if (report_out_of_range && destination_rank == 0) {
          printf(
              "direct DCP A2A: query_start_loc out of range -- token=%lld "
              "seq_idx=%lld num_seqs=%lld qsl[0]=%d (clamped)\n",
              static_cast<long long>(token_idx),
              static_cast<long long>(seq_idx),
              static_cast<long long>(num_seqs), query_start_loc[0]);
        }
        seq_idx = seq_idx < 0 ? 0 : num_seqs - 1;
      }
      empty_kv = seq_lens[seq_idx] == 0;
    }
  }
  __syncthreads();

  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  int64_t parity = static_cast<int64_t>(epoch & 1u);
  int64_t destination_item =
      ((parity * world_size + rank) * max_num_tokens + token_idx) *
      heads_per_rank;
  int64_t source_head = destination_rank * heads_per_rank;

  if (!empty_kv) {
    uint4* peer_output =
        get_peer_ptr<uint4>(peer_output_ptrs, destination_rank);
    int64_t vectors_per_item = heads_per_rank * head_dim / 8;
    int64_t source_vector =
        (token_idx * output_token_stride + source_head * head_dim) / 8;
    int64_t destination_vector = destination_item * head_dim / 8;
    for (int64_t vector_idx = threadIdx.x; vector_idx < vectors_per_item;
         vector_idx += blockDim.x) {
      peer_output[destination_vector + vector_idx] =
          partial_output[source_vector + vector_idx];
    }
  }

  float* peer_lse = get_peer_ptr<float>(peer_lse_ptrs, destination_rank);
  int64_t source_lse = token_idx * lse_token_stride + source_head;
  for (int64_t head_idx = threadIdx.x; head_idx < heads_per_rank;
       head_idx += blockDim.x) {
    peer_lse[destination_item + head_idx] =
        empty_kv ? -k_pos_inf() : to_float(partial_lse[source_lse + head_idx]);
  }
}

// Publish completion after the stream-ordered payload writes.
__global__ void signal_kernel(const int64_t* peer_signal_ptrs,
                              const int64_t* epoch_ptr, int64_t world_size,
                              int64_t rank) {
  int64_t destination_rank =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (destination_rank >= world_size) {
    return;
  }

  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  int64_t parity = static_cast<int64_t>(epoch & 1u);
  uint32_t* peer_signal =
      get_peer_ptr<uint32_t>(peer_signal_ptrs, destination_rank);
  int64_t signal_item = parity * world_size + rank;
  store_release_system(peer_signal + signal_item, epoch);
}

// Single block, one thread per source rank: the ONLY spinner in the op.
//
// Hoisting the wait out of the combine is the other half of the gfx9 spin fix
// (see kHotSpins in dcp_direct_common_hip.h). Previously every one of the
// num_tokens*heads_per_rank combine blocks ran its own polling loop, so the
// whole device sat in a storm of concurrent full-L2 invalidates for the entire
// wait. Now exactly one block polls. Still sync-free: this kernel and the
// combine below are stream-ordered, so no host round-trip is introduced.
__global__ void wait_signals_kernel(const uint32_t* received_signal,
                                    const int64_t* epoch_ptr,
                                    int64_t world_size) {
  int64_t source_rank = static_cast<int64_t>(threadIdx.x);
  if (source_rank >= world_size) {
    return;
  }
  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  int64_t parity = static_cast<int64_t>(epoch & 1u);
  int64_t signal_item = parity * world_size + source_rank;
  if (!wait_for_epoch(received_signal + signal_item, epoch)) {
    printf("direct DCP A2A timeout source=%lld epoch=%u observed=%u "
           "(proceeding with stale payload -- output for this call is INVALID)\n",
           static_cast<long long>(source_rank), epoch,
           vllm::direct_dcp::load_acquire_system(received_signal + signal_item));
    // Deliberately NOT __builtin_trap(). On AMDGPU a trap raises a queue fault,
    // which kills the worker *inside* whatever GPU allocation it was servicing.
    // Measured consequence: the driver is left releasing BOs in a bad state
    // (dmesg amdgpu_bo_release_notify via kfd_ioctl_free_memory_of_gpu), the
    // KFD eviction/restore workqueue then thrashes box-wide, and every
    // subsequent ncclCommInitRank goes from ~15 s to ~235 s -- a poisoned box
    // that a docker restart does not cure (the state is in the kernel, not the
    // container). A timeout here already means a peer is gone and the run is
    // lost; returning lets it die through the normal Python/RCCL-watchdog path
    // with the printf above naming the rank, instead of taking the driver down
    // with it.
  }
}

// Form stable base-2 LSE weights and combine in FP32. Runs only after
// wait_signals_kernel has observed every peer's release store, so it contains
// no spin at all.
template <typename scalar_t>
__global__ void lse_combine_kernel(
    const scalar_t* received_output, const float* received_lse,
    const uint32_t* received_signal, const int64_t* epoch_ptr,
    scalar_t* combined_output, int64_t world_size, int64_t num_tokens,
    int64_t max_num_tokens, int64_t heads_per_rank, int64_t head_dim,
    bool is_lse_base_on_e) {
  extern __shared__ float weights[];

  int64_t item = static_cast<int64_t>(blockIdx.x);
  int64_t token_idx = item / heads_per_rank;
  int64_t head_idx = item - token_idx * heads_per_rank;
  uint32_t epoch = static_cast<uint32_t>(epoch_ptr[0]);
  int64_t parity = static_cast<int64_t>(epoch & 1u);

  // One system-scope acquire per block (NOT in a loop) to pair with the peers'
  // release stores, so the payload they wrote is guaranteed visible here.
  if (threadIdx.x == 0) {
    (void)vllm::direct_dcp::load_acquire_system(received_signal +
                                                parity * world_size);
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    float lse_max = -k_pos_inf();
    for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
      int64_t source_item =
          ((parity * world_size + source_rank) * max_num_tokens + token_idx) *
              heads_per_rank +
          head_idx;
      float value = received_lse[source_item];
      if (isnan(value) || value == k_pos_inf()) {
        value = -k_pos_inf();
      }
      if (is_lse_base_on_e) {
        value *= K_LOG2E_F;
      }
      weights[source_rank] = value;
      lse_max = fmaxf(lse_max, value);
    }
    if (lse_max == -k_pos_inf()) {
      lse_max = 0.0f;
    }

    float lse_sum = 0.0f;
    for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
      lse_sum += exp2f(weights[source_rank] - lse_max);
    }
    float inverse_lse_sum = lse_sum > 0.0f ? 1.0f / lse_sum : 0.0f;
    for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
      weights[source_rank] =
          exp2f(weights[source_rank] - lse_max) * inverse_lse_sum;
    }
  }
  __syncthreads();

  for (int64_t dim_idx = threadIdx.x; dim_idx < head_dim;
       dim_idx += blockDim.x) {
    float accumulator = 0.0f;
    for (int64_t source_rank = 0; source_rank < world_size; ++source_rank) {
      // Empty shards leave stale payloads, so never read zero-weight sources.
      float weight = weights[source_rank];
      if (weight == 0.0f) {
        continue;
      }
      int64_t source_item =
          ((parity * world_size + source_rank) * max_num_tokens + token_idx) *
              heads_per_rank +
          head_idx;
      accumulator +=
          to_float(received_output[source_item * head_dim + dim_idx]) * weight;
    }
    combined_output[item * head_dim + dim_idx] =
        from_float<scalar_t>(accumulator);
  }
}

void direct_dcp_a2a_lse_reduce(
    const at::Tensor& partial_output, const at::Tensor& partial_lse,
    const std::optional<at::Tensor>& seq_lens,
    const std::optional<at::Tensor>& query_start_loc,
    const at::Tensor& peer_output_ptrs, const at::Tensor& peer_lse_ptrs,
    const at::Tensor& peer_signal_ptrs, at::Tensor& received_output,
    at::Tensor& received_lse, at::Tensor& received_signal, at::Tensor& epoch,
    at::Tensor& combined_output, int64_t world_size, int64_t rank,
    int64_t max_num_tokens, bool is_lse_base_on_e) {
  TORCH_CHECK(partial_output.is_cuda() && partial_lse.is_cuda(),
              "partial output and LSE must be CUDA tensors");
  auto output_dtype = partial_output.scalar_type();
  TORCH_CHECK(output_dtype == at::kHalf || output_dtype == at::kBFloat16,
              "direct DCP A2A only supports FP16 and BF16 output");
  auto lse_dtype = partial_lse.scalar_type();
  TORCH_CHECK(lse_dtype == at::kFloat || lse_dtype == at::kHalf ||
                  lse_dtype == at::kBFloat16,
              "partial LSE must be FP32, FP16, or BF16");
  TORCH_CHECK(partial_output.dim() == 3 && partial_lse.dim() == 2,
              "expected output [T,H,D] and LSE [T,H]");
  TORCH_CHECK(world_size > 1, "world_size must be greater than 1");
  TORCH_CHECK(rank >= 0 && rank < world_size, "invalid rank");

  int64_t num_tokens = partial_output.size(0);
  int64_t total_heads = partial_output.size(1);
  int64_t head_dim = partial_output.size(2);
  int64_t output_token_stride = partial_output.stride(0);
  int64_t lse_token_stride = partial_lse.stride(0);
  TORCH_CHECK(
      partial_output.stride(2) == 1 && partial_output.stride(1) == head_dim &&
          output_token_stride >= total_heads * head_dim &&
          output_token_stride % 8 == 0,
      "partial output must have packed heads and an aligned token stride");
  TORCH_CHECK(partial_lse.stride(1) == 1 && lse_token_stride >= total_heads,
              "partial LSE must have packed heads");
  TORCH_CHECK(num_tokens > 0 && num_tokens <= max_num_tokens,
              "token count exceeds symmetric buffer capacity");
  TORCH_CHECK(total_heads % world_size == 0,
              "attention heads must divide evenly across DCP ranks");
  int64_t heads_per_rank = total_heads / world_size;
  TORCH_CHECK(combined_output.scalar_type() == output_dtype &&
                  combined_output.is_contiguous() && combined_output.is_cuda(),
              "combined output must match the contiguous CUDA input");
  TORCH_CHECK(combined_output.size(0) == num_tokens &&
                  combined_output.size(1) == heads_per_rank &&
                  combined_output.size(2) == head_dim,
              "combined output has the wrong shape");
  TORCH_CHECK(received_output.scalar_type() == output_dtype &&
                  received_lse.scalar_type() == at::kFloat &&
                  received_signal.scalar_type() == at::kInt,
              "invalid symmetric staging dtypes");
  TORCH_CHECK(peer_output_ptrs.scalar_type() == at::kLong &&
                  peer_lse_ptrs.scalar_type() == at::kLong &&
                  peer_signal_ptrs.scalar_type() == at::kLong,
              "peer pointer tables must be int64");
  const int32_t* seq_lens_ptr = nullptr;
  const int32_t* query_start_loc_ptr = nullptr;
  int64_t num_seqs = 0;
  TORCH_CHECK(seq_lens.has_value() == query_start_loc.has_value(),
              "seq_lens and query_start_loc must be provided together");
  if (seq_lens.has_value() && query_start_loc.has_value()) {
    TORCH_CHECK(seq_lens->is_cuda() && seq_lens->dim() == 1 &&
                    seq_lens->stride(0) == 1 &&
                    seq_lens->scalar_type() == at::kInt,
                "seq_lens must be a contiguous 1-D int32 CUDA tensor");
    TORCH_CHECK(query_start_loc->is_cuda() && query_start_loc->dim() == 1 &&
                    query_start_loc->stride(0) == 1 &&
                    query_start_loc->scalar_type() == at::kInt,
                "query_start_loc must be a contiguous 1-D int32 CUDA tensor");
    num_seqs = seq_lens->size(0);
    TORCH_CHECK(num_seqs > 0 && query_start_loc->size(0) == num_seqs + 1,
                "query_start_loc must contain one boundary per sequence");
    seq_lens_ptr = seq_lens->data_ptr<int32_t>();
    query_start_loc_ptr = query_start_loc->data_ptr<int32_t>();
  }

  const c10::cuda::CUDAGuard device_guard(partial_output.device());
  hipStream_t stream = c10::cuda::getCurrentCUDAStream();
  constexpr int kExchangeThreads = 256;
  constexpr int kCombineThreads = 128;

  // K3_DCP_KTIME=1: per-kernel attribution. Diagnostic ONLY -- it forces a host
  // sync at the end of the call, so it defeats the sync-free design. Off by
  // default and completely out of the normal path.
  static const bool ktime = [] {
    const char* v = std::getenv("K3_DCP_KTIME");
    return v != nullptr && v[0] == '1';
  }();
  // K3_DCP_BOUNDS=1: report (once per offending token) when query_start_loc
  // does not cover a token, i.e. the seq_lens[-1] case the dispatch kernel now
  // clamps. Off by default; the clamp itself is always on. This exists so the
  // clamp can be *proven* to fire rather than silently masking the fault.
  static const bool report_oob = [] {
    const char* v = std::getenv("K3_DCP_BOUNDS");
    return v != nullptr && v[0] == '1';
  }();
  hipEvent_t ev[5] = {};
  auto stamp = [&](int i) {
    if (ktime) {
      hipEventRecord(ev[i], stream);
    }
  };
  if (ktime) {
    for (auto& e : ev) {
      hipEventCreate(&e);
    }
  }
  stamp(0);

  increment_epoch_kernel<<<1, 1, 0, stream>>>(epoch.data_ptr<int64_t>());
  check_hip_launch("direct DCP A2A");
  int64_t dispatch_blocks = world_size * num_tokens;
  auto launch_dispatch = [&]<typename lse_t>() {
    dispatch_output_lse_kernel<lse_t>
        <<<dispatch_blocks, kExchangeThreads, 0, stream>>>(
            reinterpret_cast<const uint4*>(partial_output.data_ptr()),
            reinterpret_cast<const lse_t*>(partial_lse.data_ptr()),
            seq_lens_ptr, query_start_loc_ptr,
            peer_output_ptrs.data_ptr<int64_t>(),
            peer_lse_ptrs.data_ptr<int64_t>(), epoch.data_ptr<int64_t>(),
            world_size, rank, num_tokens, max_num_tokens, num_seqs,
            heads_per_rank, head_dim, output_token_stride, lse_token_stride,
            report_oob);
  };
  if (lse_dtype == at::kFloat) {
    launch_dispatch.operator()<float>();
  } else if (lse_dtype == at::kBFloat16) {
    launch_dispatch.operator()<__hip_bfloat16>();
  } else {
    launch_dispatch.operator()<__half>();
  }
  check_hip_launch("direct DCP A2A");
  stamp(1);
  int64_t signal_items = world_size;
  int64_t signal_blocks =
      (signal_items + kExchangeThreads - 1) / kExchangeThreads;
  signal_kernel<<<signal_blocks, kExchangeThreads, 0, stream>>>(
      peer_signal_ptrs.data_ptr<int64_t>(), epoch.data_ptr<int64_t>(),
      world_size, rank);
  check_hip_launch("direct DCP A2A");
  stamp(2);
  // ONE block waits for all peers; the combine below then runs spin-free. Both
  // are on `stream`, so this stays sync-free (no host round-trip).
  wait_signals_kernel<<<1, kExchangeThreads, 0, stream>>>(
      reinterpret_cast<const uint32_t*>(received_signal.data_ptr<int32_t>()),
      epoch.data_ptr<int64_t>(), world_size);
  check_hip_launch("direct DCP A2A");
  stamp(3);
  int64_t combine_blocks = num_tokens * heads_per_rank;
  size_t shared_memory_bytes = world_size * sizeof(float);
  auto launch_combine = [&]<typename scalar_t>() {
    lse_combine_kernel<scalar_t>
        <<<combine_blocks, kCombineThreads, shared_memory_bytes, stream>>>(
            reinterpret_cast<const scalar_t*>(received_output.data_ptr()),
            received_lse.data_ptr<float>(),
            reinterpret_cast<const uint32_t*>(
                received_signal.data_ptr<int32_t>()),
            epoch.data_ptr<int64_t>(),
            reinterpret_cast<scalar_t*>(combined_output.data_ptr()), world_size,
            num_tokens, max_num_tokens, heads_per_rank, head_dim,
            is_lse_base_on_e);
  };
  if (output_dtype == at::kBFloat16) {
    launch_combine.operator()<__hip_bfloat16>();
  } else {
    launch_combine.operator()<__half>();
  }
  check_hip_launch("direct DCP A2A");
  stamp(4);

  if (ktime) {
    hipEventSynchronize(ev[4]);
    float ms[4];
    for (int i = 0; i < 4; ++i) {
      hipEventElapsedTime(&ms[i], ev[i], ev[i + 1]);
    }
    printf("[KTIME rank%lld] epoch+dispatch=%.3f signal=%.3f wait=%.3f "
           "combine=%.3f total=%.3f ms\n",
           static_cast<long long>(rank), ms[0], ms[1], ms[2], ms[3],
           ms[0] + ms[1] + ms[2] + ms[3]);
    for (auto& e : ev) {
      hipEventDestroy(e);
    }
  }
}

}  // namespace

// Extend vLLM's _C namespace so torch.ops._C.direct_dcp_a2a_lse_reduce resolves
// (the op is NOT compiled into ROCm vllm._C, so this FRAGMENT defs it fresh).
TORCH_LIBRARY_FRAGMENT(_C, m) {
  m.def(
      "direct_dcp_a2a_lse_reduce("
      "Tensor partial_output, Tensor partial_lse, Tensor? seq_lens, "
      "Tensor? query_start_loc, Tensor peer_output_ptrs, "
      "Tensor peer_lse_ptrs, Tensor peer_signal_ptrs, Tensor! received_output, "
      "Tensor! received_lse, Tensor! received_signal, Tensor! epoch, "
      "Tensor! combined_output, int world_size, int rank, "
      "int max_num_tokens, bool is_lse_base_on_e) -> ()");
}

// HIP tensors dispatch under the CUDA key in torch's ROCm build.
TORCH_LIBRARY_IMPL(_C, CUDA, m) {
  m.impl("direct_dcp_a2a_lse_reduce", &direct_dcp_a2a_lse_reduce);
}
