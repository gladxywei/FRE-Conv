# Pretrained checkpoints

[Back to the project guide](../README.md)

## Download

The weights are published in [FRE-Conv Release `checkpoints-v1`](https://github.com/gladxywei/FRE-Conv/releases/tag/checkpoints-v1). Download and install them from the repository root:

```bash
python -m freconv.download_checkpoints
```

The default release tag is recorded in [manifest.json](manifest.json); no `--release-tag` argument is needed.

The checkpoint set is named `paper-v1`. It contains one Gaussian and one complex-noise checkpoint for each of the twelve models, all trained on ICVL. The GitHub tag `checkpoints-v1` identifies the published release; it does not change the checkpoint-set ID or the local installation directory.

| Asset | Size | Download |
| --- | ---: | --- |
| `freconv-paper-v1-checkpoints.tar.gz` | 386,400,598 bytes | [Weight archive](https://github.com/gladxywei/FRE-Conv/releases/download/checkpoints-v1/freconv-paper-v1-checkpoints.tar.gz) |
| `SHA256SUMS` | 102 bytes | [Archive checksum](https://github.com/gladxywei/FRE-Conv/releases/download/checkpoints-v1/SHA256SUMS) |

The uncompressed weights total 459,500,168 bytes. Archive SHA-256:

```text
2b84622f5bd7177dc79d4041d74ac1826156b547b59b6d7f4f9acd03a3a6018c
```

The automatically generated GitHub source-code archives do not contain the weights.

## Install a matching local archive

If you downloaded `freconv-paper-v1-checkpoints.tar.gz` manually, run from the repository root (adjust the archive path if needed):

```bash
python -m freconv.download_checkpoints --archive freconv-paper-v1-checkpoints.tar.gz
```

The tool checks the archive's byte size and SHA-256, extracts into temporary storage, and verifies every expected weight's byte size and SHA-256 before installing it. It rejects path traversal and symbolic/hard links, and refuses to overwrite an existing `checkpoints/paper-v1/` directory. Only load checkpoints from a trusted source; checksum matching checks file identity, not whether a serialized checkpoint is safe to execute.

Verify an existing installation without downloading or extracting anything:

```bash
python -m freconv.download_checkpoints --verify-only
```

For example, the two `fman_t4` checkpoints are installed as:

```text
checkpoints/
  paper-v1/
    fman_t4/
      fman_t4-hsidenoising-epoch-80-l2.pth
      fman_t4-hsidenoising-epoch-130-l2.pth
```

These are the Gaussian and complex-noise weights, respectively. Paths for all models are listed under `target_path` in [manifest.json](manifest.json).

## Configuration reference

| File | Role |
| --- | --- |
| [manifest.json](manifest.json) | Archive metadata and 24 per-file identities: model, task, target path, bytes, and SHA-256. |
| [registry.yaml](registry.yaml) | Evaluation model groups and the mapping from each model/task to a checkpoint path. |

`paper12` selects all models. `baseline_t4` selects the four standard-convolution models and their T4 counterparts. The evaluation entrypoints use the `paper-v1` set by default and select task weights according to the noise mode; Washington DC Mall uses the requested `--task`.
