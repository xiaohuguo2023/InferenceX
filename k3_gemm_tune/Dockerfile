# Kimi-K3 bf16 (a16w16) GEMM tuning image — MI355X / gfx950
#
# Bakes the EXACT aiter build that serves K3 (commit 00cbe979f, local/unpushed) so the
# tuned libtype/solidx match at serve time. Self-contained: build once, ship to any
# MI355X (docker save/load or push), run to produce kimik3_bf16_tuned_gemm.csv.
#
# Build context must contain:
#   aiter/                          (= ~/work/aiter @ 00cbe979f, with prebuilt hsa/*.co)
#   kimik3_bf16_tuning_gemm.csv     (the 424-row curated shape list)
#   tune.sh
# Use build.sh, which stages this context for you.
FROM vllm/vllm-openai-rocm:kimi-k3

ENV GPU_ARCHS=gfx950 \
    PREBUILD_KERNELS=0 \
    AITER_USE_SYSTEM_TRITON=1

# 1) exact aiter source (includes the prebuilt gfx950 .co kernels)
COPY aiter /opt/aiter

# 2) editable install (CPU-only at build time); strip the stale committed jit .so so the
#    GPU-side JIT rebuild of module_aiter_core happens on first run (tune.sh), matching
#    the Aug source (MlaVersion etc.). No GPU is available during docker build.
RUN cd /opt/aiter \
 && git config --global --add safe.directory '*' 2>/dev/null || true \
 && pip uninstall -y aiter amd-aiter >/dev/null 2>&1 || true \
 && rm -f aiter/jit/*.so && rm -rf aiter/jit/build \
 && pip install -e . --no-build-isolation --no-deps

# 3) tuning input (baked) + driver
COPY kimik3_bf16_tuning_gemm.csv /opt/tune_input/kimik3_bf16_tuning_gemm.csv
COPY tune.sh /usr/local/bin/tune.sh
RUN chmod +x /usr/local/bin/tune.sh

WORKDIR /work
ENTRYPOINT ["/usr/local/bin/tune.sh"]
