# The standalone tower

The merged layout (`rv/merge.py`) widens the host model and puts the RV32 machine in the added
residual dims. Some hosts cannot be widened: a ternary body whose packing does not keep zero
padding word for word, or a serving engine whose kernels exist for one hidden size. For those,
the machine is exported as a tower of its own.

The tower is the same circuit, compiled for the host's Gated DeltaNet head geometry (key heads,
value heads, MLP width), with its own residual stream, token table and states. The host's weights
are not read and not changed. A decoder runs every token through the tower and reads the output
code; text tokens also run through the host. This is what the fast path of `rv/chat.py` already
does with a merged checkpoint (`RvChat._build_fast`).

Model ids are the merged layout's: a host token keeps its id, chip token `t` is `vocab + t`.

## Export: `rv/export_tower.py`

    python -m chipformer.rv.export_tower HOST --out dist/tower-27b \
        [--firmware tests/data/rv/calc-bridge.elf]

`HOST` is a directory with the host's `config.json` and `tokenizer.json`.

Written files:

- `model.safetensors`
  - The chip layers under their Hugging Face names, bf16.
  - `model.embed_tokens.weight`: the host's text rows (their chip features only), then the chip
    tokens, then zero padding rows.
  - With `--firmware`, `firmware.NAME.state`: the parked firmware as the DeltaNet states, shape
    (layers, value heads, key dim, value dim), fp32. It is what `hf.inject` writes into the
    Hugging Face cache.
- `config.json`: architecture `ChipTower`, the tower's text config, and `chip_config`, which holds
  the layout (the fields of `rv/merge.py`'s `chipformer.json`) and the firmware names.

The MLPs are cut to the units the circuit uses. Units are emitted from unit 0 up, so the rest of
the host's width is zero and is dropped; the export asserts this.

`load_reference(path)` returns the tower in stock Hugging Face modules. It appends one zero
full-attention layer, which the Hugging Face cache needs to count positions.
`hidden_states[chip_layers]` is the residual after the chip layers.

For Qwen3.8-27B's geometry (16 key heads, 48 value heads, 17408 MLP units) the circuit has 11
layers, the tower's hidden size is 1280 with 5 ballast dims, and the MLPs keep 8576 units.

## Check: `rv/towercheck.py`

    python -m chipformer.rv.towercheck TOWER HOST --firmware tests/data/rv/calc-bridge.elf \
        [--mode decode|prefill] [--chunk N] [--served] [--device cpu|mps|cuda] \
        [--exprs ...] [--random N --seed S] [--golden out.parquet] [--control rom|weight|fp16]

The check needs no host weights: the host's text is teacher-forced. The stream is the one a
decoder would produce: the chat prompt with the calc tool, then for each expression the host's
tool call, the machine's run with its echoes, and a few text tokens.

At every position:

- the residual after the chip layers must equal the IR evaluator's on the kept dims (output code,
  tau, mu), bit for bit;
- the output code must be set exactly where the reference machine owns the next token;
- where it is set, it must name that token.

`--served` runs the prompt from zero states and loads the firmware state after it, as a decoder
does. Without it the state is loaded first, as the IR has it.

`--mode decode` runs token by token. `--mode prefill` runs chunks of `--chunk` tokens through the
chunked delta rule.

Controls. Each must fail, and with `--control` the exit status is 0 only if the check fails:

- `rom` flips the low bit of each byte of the ROM word the parked machine runs next.
- `weight` flips the sign of one MLP output weight.
- `fp16` keeps the DeltaNet states in fp16. A state row's ballast, 2^24, does not fit.

`--golden out.parquet` writes the stream and the kept values, for another runtime to be held to
(extra `golden`):

- `out.parquet`, zstd: `position`, `model_id`, `owned` and one `dim_<residual index>` column per
  kept dim. The schema metadata holds `inject_at`, the position where the firmware state is
  loaded (the prompt length with `--served`).
- `out.raw/`, for a runtime without a parquet reader: `ids.i32` (little-endian int32 model ids),
  `kept.f32` (little-endian float32, positions by kept dims, row-major) and `golden.json`
  (positions, the kept dims in column order, `inject_at`).

The golden rows include the prompt positions.

Devices. CPU bf16 is correct but far too slow for decode mode. MPS and CUDA are bit-exact. On
MPS, a long prefill run can have a command buffer aborted by the system
(`kIOGPUCommandBufferCallbackErrorImpactingInteractivity`); such a run is void.

## Ternary planes: `rv/ternary.py`

    python -m chipformer.rv.ternary TOWER [--group 128] [--max-planes 4] [--out FILE] [--control]

A plane is a ternary matrix `T` with weights in {-1, 0, +1} and one scale per row and group of
`group` columns. A matrix is

    W = sum over p of scale_p[row, col // group] * T_p[row, col]

Each weight sits in exactly one plane. Within a group the distinct magnitudes are ranked, and
plane `p` holds the weights of rank `p`, with that magnitude as the group's scale. So the sum has
one nonzero term per weight and equals `W` bit for bit. A matmul over the planes adds the same
products as the dense one, with fp32 accumulation, in another order.

The plane count of a matrix is the largest number of distinct magnitudes in any of its groups.

The scale may run the other way: one per column and group of rows, which is the planes of the
transposed matrix (`scale_axis` "column"). It is used where it needs fewer planes, as in the MLP
output maps, whose columns are units with one coefficient each.

The scales are the circuit's own weights, so they are bf16 values. The report says whether they
also fit binary16.

Without `--out` only the table is printed and `ternary.json` is written. With it, the planes are
written (trits int8, scales bf16; keys end in `.trits` and `.scales`, or `.tritst` and `.scalest`
for the column axis) for every matrix with at most `--max-planes` planes. The others, and the
tensors that are not matrices, stay dense.

`--control` flips one sign in the first plane set. Its rebuild must differ, and the exit status
is 0 only if it does.

On the 27B tower: 82 of 89 matrices need at most two planes and none needs more than seven. The
scales range from 2^-41 to 2^125, so they do not fit binary16.
