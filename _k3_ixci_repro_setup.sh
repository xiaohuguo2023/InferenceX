#!/usr/bin/env bash
# Recreate the pinned ROCm 7.2.3 K3 repro container on this node.
# Run ON THE GPU NODE (via srun), not on the login node.
set -euo pipefail

NODE_TAG="${1:?node tag e.g. 270}"
IMAGE="${IMAGE:-vllm/vllm-openai-rocm:nightly-385dce36bcee42309924a5ece951a96db3dce7f2}"
NAME="k3-repro-385dce-n${NODE_TAG}"

# The pinned nightly tag has been rotated off the registry, so a cached copy on
# the node is the only way to get it. Only pull when it is not already here.
if docker image inspect "$IMAGE" >/dev/null 2>&1; then
  echo "=== using cached $IMAGE ==="
else
  echo "=== pull $IMAGE ==="
  docker pull "$IMAGE"
fi

# Slurm gives us the node exclusively, but containers from finished jobs survive
# the job that started them and keep holding VRAM, which later shows up as an
# engine-init OOM. Reap everything before claiming the GPUs.
echo "=== reap leftover containers ==="
docker ps -a --format '{{.Names}} {{.Status}}' || true
if [ "${SKIP_REAP:-0}" = 1 ]; then
  # Shared node: replace only our own container, leave the others running.
  docker rm -f "$NAME" >/dev/null 2>&1 || true
else
  docker ps -aq | xargs -r docker rm -f >/dev/null 2>&1 || true
fi
sleep 5

# The driver releases VRAM a few seconds after teardown; fail loudly rather than
# letting the engine start on a node that is still occupied.
free_mib=$(rocm-smi --showmeminfo vram --csv 2>/dev/null |
  awk -F, 'NR>1 && $3>0 {u=$3/1048576; if (u>max) max=u} END {printf "%d", max}')
echo "=== max VRAM in use after reap: ${free_mib:-unknown} MiB/GPU ==="
if [ -n "${free_mib:-}" ] && [ "$free_mib" -gt 2048 ]; then
  echo "WARNING: ${free_mib} MiB/GPU still held after reap; check for stray processes" >&2
fi

# --group-add resolves names against the IMAGE's /etc/group, not the host's.
# The rocm100 nightlies ship without `render`, so pass host GIDs numerically.
GID_VIDEO="$(getent group video | cut -d: -f3)"
GID_RENDER="$(stat -c %g /dev/kfd 2>/dev/null)"
: "${GID_VIDEO:=44}" "${GID_RENDER:=992}"

echo "=== run $NAME (video=$GID_VIDEO render=$GID_RENDER) ==="
docker run -d --name "$NAME" --entrypoint sleep \
  --device=/dev/kfd --device=/dev/dri \
  --network=host --ipc=host \
  --group-add "$GID_VIDEO" --group-add "$GID_RENDER" --group-add 10008 \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  --shm-size=128g \
  -v /home/xiaohugu/work/InferenceX_dcp:/inferencex_dcp \
  -v /home/xiaohugu/work/InferenceX:/workspace \
  -v /home/xiaohugu:/home/xiaohugu \
  -v /shared_nfs:/shared_nfs \
  -v /it-shared:/it-shared \
  -v /dev/shm:/dev/shm \
  -e HF_HUB_CACHE=/shared_nfs/hf-hub-cache \
  -e HF_HOME=/shared_nfs/hf-hub-cache \
  -e GPU_ARCHS=gfx950 \
  "$IMAGE" infinity

echo "=== container ==="
docker ps --filter "name=$NAME" --format '{{.Names}} {{.Status}} {{.Image}}'
docker exec "$NAME" bash -lc 'cat /opt/rocm/.info/version; python3 -c "import vllm; print(vllm.__version__)"'
echo "SETUP_OK $NAME"
