# Humanoid-GPT 264

[Author project](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT) · [Checkpoint](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT/blob/9f9e7b74ecadb532abbb34b6a779d87191a9bbb6/storage/ckpts/pns_wo_priv264.onnx) · [Provenance](provenance.json)

## Install and run

```bash
git clone --recurse-submodules https://github.com/MotionChallenge-Project/MotionChallenge-humanoid-gpt.git
cd MotionChallenge-humanoid-gpt
uv sync
uv run evaluate.py
```

Produces success, completion and six MPBPE values in a verified `result.json`.
MuJoCo 3.17 CPU batch; `--jobs` sets parallelism. Missing data and weights are
downloaded and hash-checked; results and traces stay outside Git.

[Inference validation](validation/README.md) · [Native parity report](validation/native-parity.json)

## Observations

Ideal proprioception, without robot world-position or contact feedback. [Checkpoint requirements](admission.json) · [Rules](https://github.com/MotionChallenge-Project/MotionChallenge-Core/blob/da0aec4023bbc3d29be67e1fce9fa9ab8a48c4b5/docs/OBSERVATIONS.md).

Author infer_onnx_real hardware route: fixed synthetic root [0,0,0.78], author reference FK, EMA and action history. No measured robot position or position integration.
