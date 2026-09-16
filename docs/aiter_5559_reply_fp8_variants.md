The premise does not hold: `get_mla_metadata_v1` does not map both e4m3 variants
to `AITER_DTYPE_fp8`. The torch->aiter dtype map is built per process from the
running arch (`aiter/utility/dtypes.py`: `fp8 = get_dtype_fp8()` ->
`defaultDtypes[get_gfx_runtime()]["fp8"]`), so it holds exactly one e4m3 key.

On gfx950:

```python
>>> from aiter.utility.dtypes import _torch_to_aiter_dtype, _aiter_dtype_id
>>> [str(d) for d in _torch_to_aiter_dtype if "float8" in str(d)]
['torch.float8_e4m3fn', 'torch.float8_e8m0fnu']
>>> _aiter_dtype_id(torch.float8_e4m3fnuz)
AssertionError: Unsupported dtype: torch.float8_e4m3fnuz
```

`defaultDtypes` maps gfx942 -> `float8_e4m3fnuz` and gfx950 -> `float8_e4m3fn`;
`AITER_DTYPE_fp8` means "this arch's fp8". `get_mla_metadata_v1` routes every
dtype through `_aiter_dtype_id`, so a gfx950 fnuz pair raises before any kernel
runs -- it cannot reach the planner and produce an undersized buffer.

Widening the predicate to `in (float8_e4m3fn, float8_e4m3fnuz)` would report fp8
for a dtype that arch cannot execute, and would then select the fp8
native-support clauses for it. Comparing against `dtypes.fp8` tracks the same
per-arch resolution the rest of aiter uses, which is the intended semantic.

Happy to add a regression test for the arch-mismatch path if you would like one,
though it would assert the `AssertionError` rather than a sizing result.
