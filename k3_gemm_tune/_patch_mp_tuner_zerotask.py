#!/usr/bin/env python3
"""Let --shape_grouped survive shapes that produce no candidate kernels.

mp_tuner groups tasks by info_keys and assumes one group per shape, asserting
len(task_group) == len(in_datas). A shape whose candidates are all rejected
(e.g. every flydsl tiling is invalid for M=16817) contributes zero tasks, so the
group count comes up short and the assertion kills the whole batch -- on K3 that
cost 10 shapes per crash and left the shard reporting success with 0 tuned.

in_datas already carries the per-shape task count (tasks_data.append((n, ()))),
so the shape index for each group is recoverable from the cumulative offsets --
the same mapping the non-grouped branch does with np.searchsorted. Fall back to
that when the counts disagree and keep the 1:1 fast path when they don't.

Applied from tune.sh when AITER_LIVE_MOUNT=1.
"""
from __future__ import annotations

import sys
from pathlib import Path

AITER = Path(sys.argv[1] if len(sys.argv) > 1 else "/opt/aiter")
MP_TUNER_PY = AITER / "aiter/utility/mp_tuner.py"

MARKER = "PATCH(mp-tuner-zerotask): tolerate shapes with no candidate kernels"

OLD = """        info_key_groups = OrderedDict()
        for task in tasks:
            info_keys = task[0][0] if task and len(task) > 0 else None
            if info_keys not in info_key_groups:
                info_key_groups[info_keys] = []
            info_key_groups[info_keys].append(task)

        task_group = list(info_key_groups.values())
        print(
            f"[Task Grouping] Grouped {len(tasks)} tasks into {len(task_group)} groups by info_keys"
        )

        # in_datas already has one entry per shape from the tuner;
        # just verify cardinality matches and use it directly.
        assert len(task_group) == len(
            in_datas
        ), f"shape_grouped: group count ({len(task_group)}) != in_datas count ({len(in_datas)})"
        ref_data_index = list(range(len(task_group)))
"""

NEW = f"""        info_key_groups = OrderedDict()
        group_first_task = {{}}
        for _task_idx, task in enumerate(tasks):
            info_keys = task[0][0] if task and len(task) > 0 else None
            if info_keys not in info_key_groups:
                info_key_groups[info_keys] = []
                group_first_task[info_keys] = _task_idx
            info_key_groups[info_keys].append(task)

        task_group = list(info_key_groups.values())
        print(
            f"[Task Grouping] Grouped {{len(tasks)}} tasks into {{len(task_group)}} groups by info_keys"
        )

        # {MARKER}
        # in_datas has one entry per shape; a shape whose candidates were all
        # rejected contributes zero tasks, so groups are not 1:1 with in_datas.
        if len(task_group) == len(in_datas):
            ref_data_index = list(range(len(task_group)))
        else:
            import numpy as np

            _cumulative = np.cumsum([size for size, _ in in_datas])
            ref_data_index = np.searchsorted(
                _cumulative,
                [group_first_task[k] for k in info_key_groups],
                side="right",
            )
            print(
                f"[Task Grouping] {{len(in_datas) - len(task_group)}} shape(s) produced no "
                f"candidate kernels; remapped {{len(task_group)}} groups onto in_datas"
            )
"""


def main() -> None:
    text = MP_TUNER_PY.read_text()
    if MARKER in text:
        print(f"already patched: {MP_TUNER_PY.name} ({MARKER})")
        return
    if OLD not in text:
        raise SystemExit(f"patch anchor missing in {MP_TUNER_PY}: {MARKER}")
    MP_TUNER_PY.write_text(text.replace(OLD, NEW, 1))
    print(f"patched OK: {MP_TUNER_PY.name} ({MARKER})")


if __name__ == "__main__":
    main()
