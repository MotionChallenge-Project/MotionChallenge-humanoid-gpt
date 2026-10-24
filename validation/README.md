# Native inference validation

`native-parity.json` (macOS) and `native-parity-linux.json` compare the adapter
with the pinned original inference over 160 states, four resets and 32 endpoint
cases. Both pass the fixed `2e-5` absolute tolerance. The reports disclose
platform-specific differences; they do not claim universal bitwise equivalence.

```bash
uv run validate_inference.py --source /path/to/author-source --weights /path/to/pns_wo_priv264.onnx --output /outside/repository/parity.json
```

`native-sources.json` pins the author files. The comparison covers reference
filtering, feature kinematics, observations, actions, history and motor targets.

`checkpoint-graph.json` describes the released 264 MLP. The separate
`pns_wo_priv216_trans.onnx` Transformer is not evaluated by this adapter.
