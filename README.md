# Motion Challenge · humanoid-gpt

[humanoid-gpt](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT) checkpoint adapter. G1 29DoF; LAFAN40 is the first suite.

## Install and run

```sh
git clone --recurse-submodules https://github.com/motion-challenge-project/motion-challenge-humanoid-gpt.git
cd motion-challenge-humanoid-gpt
uv sync
uv run evaluate.py
```

Reuses local `checkpoints/policy.onnx` or the external cache; missing artifacts are
fetched and hash-checked. Use `--weights PATH --data PATH --offline` for local files.
Results and traces are written outside Git. `--help` lists options.

The root `motion-challenge` submodule owns data, physics, metrics and verification.
This repository only adapts model inference; no `sim2real` dependency.

[Provenance](provenance.json)

MuJoCo 3.17 batch is the default. Use `--jobs 8` for parallel clips;
download and evaluation progress updates every 5 seconds.


## Checkpoint source

[GalaxyGeneralRobotics/Humanoid-GPT](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT/blob/9f9e7b74ecadb532abbb34b6a779d87191a9bbb6/README.md) · [pns_wo_priv216.onnx](https://github.com/GalaxyGeneralRobotics/Humanoid-GPT/blob/9f9e7b74ecadb532abbb34b6a779d87191a9bbb6/storage/ckpts/pns_wo_priv216.onnx). Author-released checkpoint.

The adapter pins this published artifact by SHA256. Artifact identity is separate from inference parity.

One `uv run evaluate.py` produces one verified `result.json`: success, completion,
three MPBPE alignments through first failure and the same three over the full motion.
Both scopes use the same checkpoint and source revision; no branch switching is needed.
