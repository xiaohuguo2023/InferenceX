"""Stop the recipe, vLLM serve, LMCache server and aiperf inside the container."""
import os
import signal
import subprocess
import time

PATTERNS = (
    "vllm serve",
    "lmcache server",
    "aiperf",
    "k3_ix_c1_",
    "_k3_ixci_repro_run.sh",
    "kimik3_fp4_mi355x_mtp.sh",
    "spawn_main",
    "VLLM::",
)

me = os.getpid()
out = subprocess.check_output(["ps", "-eo", "pid,cmd"], text=True)
pids = []
for line in out.splitlines()[1:]:
    pid_s, _, cmd = line.strip().partition(" ")
    pid = int(pid_s)
    if pid in (me, 1) or "k3_stop_serve.py" in cmd:
        continue
    if any(p in cmd for p in PATTERNS):
        pids.append(pid)
print("stopping", pids)
for p in pids:
    try:
        os.kill(p, signal.SIGTERM)
    except ProcessLookupError:
        pass
time.sleep(15)
for p in pids:
    try:
        os.kill(p, signal.SIGKILL)
    except ProcessLookupError:
        pass
print("stopped")
