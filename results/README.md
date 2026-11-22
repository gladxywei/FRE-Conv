# Results and metric reference

[Back to the project guide](../README.md)

These tables report the four backbones with standard convolution (`Conv`), eight-rotation FRE-Conv (`T8`), and four-rotation FRE-Conv (`T4`). They provide the manuscript's ICVL results and the Chikusei zero-shot evaluation results.

## Files and aggregation

| File | Rows | Contents |
| --- | ---: | --- |
| [icvl_paper_results.csv](icvl_paper_results.csv) | 108 | Twelve models × nine ICVL noise modes; parameters and metrics transcribed from the manuscript tables. |
| [chikusei_per_mode.csv](chikusei_per_mode.csv) | 108 | Twelve models × nine Chikusei modes; each row averages eight fixed crops. |
| [chikusei_aggregate.csv](chikusei_aggregate.csv) | 12 | One row per Chikusei model, averaging 72 observations: nine modes × eight crops. |

The modes are Gaussian `30`, `50`, `70`, and `blind`, and complex `case1` through `case5`. Model IDs match the [model registry](../freconv/models/registry.py). In the Chikusei aggregate file, `Baseline` denotes the same variant labeled `Conv` in the other files.

## Columns and metrics

- `model`, `backbone`, and `variant` identify the network; not every file repeats all three fields.
- `mode` identifies the noise setting in a per-mode file.
- `params_m` is the manuscript's learnable parameter count in millions.
- `scenes` is the number of crops averaged in a Chikusei model-mode row (8).
- `observations` is the total number of crop-mode observations per Chikusei model (72).
- `psnr` is PSNR in dB; higher is better.
- `ssim` is structural similarity; higher is better.
- `sam` is spectral angle in radians; lower is better.

For Chikusei comparisons with new evaluation output, the table's `psnr` corresponds to the evaluator's `my_psnr` (mean bandwise PSNR), not its separate whole-cube `psnr` column. The evaluator also computes mean bandwise SSIM and spatially averaged spectral angle. See [evaluate.py](../freconv/evaluate.py) for the metric implementation.

The project README's ICVL summary is an arithmetic mean over nine rounded manuscript values for each model. Chikusei summaries are rounded from the model aggregates. These summaries are not pooled-pixel PSNR and do not imply that every individual noise mode improves.

## Result sources

The ICVL table is a normalized transcription of the twelve model columns in the manuscript's Gaussian- and complex-noise tables. Parameter counts come from the same tables. The manuscript source revision is `9d7e1d093c5177451680f7c8ced862ac02477527`.

The Chikusei per-mode table was computed from the scene-level experiment records. Each row is an arithmetic mean over eight fixed crops. The Conv/T4 experiments used implementation revision `c718281f0980a474891718042373ea3a8f4e4247`; the T8 experiments used `f92e11757d12b9eaa192b1530b08bd589cb7d6ac`.

## Evaluation setting

Chikusei uses synthetic noise on pseudo-clean crops, with ICVL-trained weights and no target-dataset adaptation. Washington DC Mall provides qualitative real-sensor evaluation without a paired clean reference; its outputs are visualizations and no-reference diagnostics rather than PSNR, SSIM, or SAM scores.
