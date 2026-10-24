# Native inference comparison

`validate_inference.py` checks the original author functions at revision
`9f9e7b74ecadb532abbb34b6a779d87191a9bbb6`, verified against `native-sources.json`.
It executes the original EMA, qpos conversion, state update, observation,
CPU ONNX policy and motor-target mapping on identical sequences.

```bash
git clone https://github.com/GalaxyGeneralRobotics/Humanoid-GPT.git /tmp/Humanoid-GPT
git -C /tmp/Humanoid-GPT checkout 9f9e7b74ecadb532abbb34b6a779d87191a9bbb6
uv run validate_inference.py --source /tmp/Humanoid-GPT \
  --weights /tmp/Humanoid-GPT/storage/ckpts/pns_wo_priv264.onnx \
  --output /tmp/humanoid-gpt-native-parity.json
```

160 states, 4 resets, 32 endpoint cases, shuffled joints and quaternion sign flips
pass the fixed 2e-5 absolute tolerance with maximum observed differences 0.
Each implementation maintains its own previous-action state.

The author reference conversion uses the 5010 inertial model; the adapter retains
that model only for feature computation. Mesh geometry is removed from the FK
asset, with every body transform and explicit inertial preserved and checked.
The shared benchmark continues to use its canonical robot, unfiltered scoring
reference, resets and failure rules. Reference packets arrive at 50 Hz; the adapter
applies the author's causal EMA, holds the last feature at the endpoint, and
uses a fresh inference instance for each motion. These differences from loading
an author NPZ or running multiple clips in one viewer are recorded in provenance.

This comparison establishes inference equivalence on the tested sequences;
complete benchmark results still require independent trace verification and a
repeat run. No prior 216 scores are transferred to 264.
