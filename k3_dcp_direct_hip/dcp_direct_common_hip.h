// SPDX-License-Identifier: Apache-2.0
// Native HIP port of vLLM's csrc/libtorch_stable/attention/dcp_utils/
// dcp_direct_common.cuh  (hand-rewritten for ROCm/MI355X gfx950 -- NOT hipify).
//
// Only the cross-rank ordering primitives differ from the CUDA original:
//   CUDA  st.global.release.sys.u32 / ld.global.acquire.sys.u32  (inline PTX)
//   HIP   __hip_atomic_store/load @ __HIP_MEMORY_SCOPE_SYSTEM     (builtins)
// Both proven equivalent + cross-GPU coherent on this box by the Step-0 spike
// (_dcp_coherence_spike.*: 20 epochs x 8 ranks, 0 timeouts, 0 mismatches).
// The CUDA-only multimem/multicast helpers are intentionally DROPPED -- the a2a
// LSE-reduce op never used them (they belonged to a separate all-reduce op).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>
#include <string>

#include <torch/library.h>

namespace vllm {
namespace direct_dcp {

// gfx9 spin-wait hygiene (MI355X, 2026-08-20).
// load_acquire_system() compiles to
//     global_load_dword v, v, s[] sc0 sc1
//     buffer_inv sc0 sc1          <-- invalidates the WHOLE L1+L2
// so every poll of a naive tight spin nukes the device cache hierarchy. The
// CUDA original is safe here because ld.acquire.sys does not invalidate; the
// port inherited that pattern unexamined. Hence: poll hot only a few times (the
// signal is normally already published), then back off with s_sleep so the
// invalidate rate collapses. Paired with hoisting the wait into its own
// single-block kernel so ONE block spins rather than
// num_tokens*heads_per_rank of them.
//
// HONESTY NOTE: this is defensive hygiene, NOT a measured win. It was first
// committed believing it explained a ~1193 ms/call cost; after the fix the cost
// was ~1100 ms, i.e. unchanged. That cost was then traced to a box-level stall
// entirely outside this op (a plain matmul loop stalls identically once RCCL is
// initialised, before this .so is even loaded). Keep the backoff -- it is
// strictly cheaper than the tight spin and costs nothing when the signal is
// already there -- but do not cite it as the fix for any latency number.
constexpr uint32_t kHotSpins = 8;
// s_sleep quantum is 64 clocks; 32 => ~1 us of backoff per poll at ~2 GHz.
constexpr int kSleepQuanta = 32;
// Bound the backed-off wait at roughly 8 s before declaring a timeout.
constexpr uint64_t kSpinLimit = 8000000;

// Advance the invocation ID; its low bit selects one of two staging slots.
static __global__ void increment_epoch_kernel(int64_t* epoch) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    epoch[0] += 1;
  }
}

template <typename T>
__device__ __forceinline__ T* get_peer_ptr(const int64_t* peer_ptrs,
                                           int64_t peer) {
  return reinterpret_cast<T*>(static_cast<uintptr_t>(peer_ptrs[peer]));
}

// System-scope release store: publish prior writes, then post the flag so a
// peer's acquire-load observes both. HIP builtin == the CUDA st.release.sys.u32.
__device__ __forceinline__ void store_release_system(uint32_t* ptr,
                                                     uint32_t value) {
  __hip_atomic_store(ptr, value, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}

// System-scope acquire load: pair with a peer's release store to see its data.
__device__ __forceinline__ uint32_t load_acquire_system(const uint32_t* ptr) {
  return __hip_atomic_load(const_cast<uint32_t*>(ptr), __ATOMIC_ACQUIRE,
                           __HIP_MEMORY_SCOPE_SYSTEM);
}

// Has the peer reached AT LEAST `epoch`? Monotonic, not exact-match.
//
// The signal slot is selected by parity (epoch & 1) but stores the full epoch,
// so slot p successively holds E, E+2, E+4, ... An `== epoch` test therefore
// has no recovery path: the instant a peer publishes E+2 into the slot we are
// still waiting on E for, the condition can never be satisfied again and the
// kernel spins until the timeout. Under lockstep call counts that overtake is
// unreachable (a peer can only publish E+2 after our combine(E) has retired),
// but "unreachable under an assumption we cannot enforce across ranks" is
// exactly how the warmup deadlock presented. `>=` costs one subtraction and
// turns that class of skew from a permanent hang into a benign early exit.
//
// The subtraction is done in int32 so it stays correct across the uint32 wrap:
// (observed - epoch) is the signed distance, negative iff the peer is behind.
__device__ __forceinline__ bool epoch_reached(uint32_t observed,
                                              uint32_t epoch) {
  return static_cast<int32_t>(observed - epoch) >= 0;
}

__device__ __forceinline__ bool wait_for_epoch(const uint32_t* ptr,
                                               uint32_t epoch) {
  // Hot phase: the peer has usually already signalled by the time we get here.
  for (uint32_t hot = 0; hot < kHotSpins; ++hot) {
    if (epoch_reached(load_acquire_system(ptr), epoch)) {
      return true;
    }
  }
  // Backed-off phase: each poll costs a full cache invalidate, so sleep between
  // them instead of thrashing L2 for the whole wait.
  for (uint64_t spins = 0; spins < kSpinLimit; ++spins) {
    __builtin_amdgcn_s_sleep(kSleepQuanta);
    if (epoch_reached(load_acquire_system(ptr), epoch)) {
      return true;
    }
  }
  return false;
}

inline void check_hip_launch(const char* operation) {
  hipError_t error = hipGetLastError();
  TORCH_CHECK(error == hipSuccess,
              std::string(operation) +
                  " kernel launch failed: " + hipGetErrorString(error));
}

}  // namespace direct_dcp
}  // namespace vllm
