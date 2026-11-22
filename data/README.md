# Datasets

[Back to the project guide](../README.md)

The loaders use float32 `.npy` cubes with shape `[bands, height, width]`. Dataset locations, crop sizes, normalization settings, and expected file counts are defined in [registry.yaml](registry.yaml). Run the preparation commands from the repository root; replace the example input filenames with your local files.

| Dataset | Role | Prepared location | Default evaluation crop |
| --- | --- | --- | --- |
| ICVL | Supervised training and synthetic-noise testing | `data/ICVL/{train,vval,test}/` | 512 × 512 |
| Chikusei | Zero-shot synthetic-noise testing | `data/Chikusei/test/` | Eight fixed 512 × 512 crops |
| Washington DC Mall | Qualitative real-sensor testing, without clean ground truth | `data/WDC_Mall/real/` | 256 × 256 center crop |

The dataset providers' access and usage terms apply independently of this software. The preparation tools operate on locally obtained data; they do not download datasets.

## ICVL

Download the **31-band MAT cubes** from the [official ICVL dataset page](https://icvl.cs.bgu.ac.il/pages/researches/hyperspectral-imaging.html) or its [Hugging Face mirror](https://huggingface.co/datasets/ICVL-BGU/ICVL_HS_2016). On the mirror, select the files under `mat/`; sign in and accept the dataset access conditions when prompted. These cubes cover 400–700 nm at 10 nm intervals. The 519-band `raw/` files and RGB `preview/` images are not the inputs used by this pipeline.

The MAT files contain a radiance array named `rad` and a wavelength vector named `bands`. Save the radiance cube as a float32 `.npy` array with the 31-band axis first: `[31, height, width]`. Inspect the array shape when reading MAT/HDF5 files, since readers may expose the axes in different orders. Preserve the spectral order and radiance values; do not convert to RGB or uint8, and do not add synthetic noise during conversion. The loaders perform normalization and noise generation.

Place the converted arrays into these directories:

```text
data/ICVL/
  train/*.npy
  vval/*.npy
  test/*.npy
```

Select 100 training cubes, 10 validation cubes in `vval`, and 50 test cubes, with no scene shared between splits. You can randomly select these disjoint subsets; record the scene filenames and split seed for repeatable experiments. Results on a newly selected split may differ from the manuscript's reported values.

Inspect the existing layout:

```bash
python -m freconv.data.prepare icvl --root data/ICVL
```

This command checks for three-dimensional arrays and reports their counts and shapes. Compare the report with the split sizes above, and ensure that arrays are bands-first float32. Training uses random 64 × 64 crops; the configured validation and test crop is 512 × 512. Loaders normalize each spectral band before adding synthetic noise.

## Chikusei

The registered source is the [Chikusei dataset page](https://naotoyokoya.com/Download.html), with a [dataset mirror](https://huggingface.co/datasets/danaroth/chikusei) also recorded in the registry. The preparation command expects the ENVI cube `HyperspecVNIR_Chikusei_20140729.bsq` and its `.hdr` header:

```bash
python -m freconv.data.prepare chikusei \
  --input data/raw/HyperspecVNIR_Chikusei_20140729.bsq \
  --header data/raw/HyperspecVNIR_Chikusei_20140729.hdr
```

The tool verifies both source SHA-256 digests against the recorded inputs, reads the ENVI layout, applies the header's reflectance scale when present, and saves eight fixed coverage crops in `data/Chikusei/test/`. A generated `data/Chikusei/manifest.json` records source checksums and crop coordinates. The default crop size and verified source files define the protocol used by the result tables; different inputs or crop settings are not the same benchmark.

The cube has 128 spectral bands. All eight crops are evaluated with ICVL-trained weights; no Chikusei training or adaptation is performed by the evaluation entrypoint.

## Washington DC Mall

The registered source is the [MultiSpec hyperspectral data page](https://engineering.purdue.edu/~biehl/MultiSpec/hyperspectral.html). The configuration expects the 191-band Washington DC Mall cube. Convert a locally obtained MAT, NPY, or NPZ cube:

```bash
python -m freconv.data.prepare wdc_mall \
  --input data/raw/wdc_mall.mat --key your_array_key
```

Replace `your_array_key` with the array name in the MAT or NPZ file; omit `--key` for an NPY file. Use `--bands-axis 0`, `1`, or `2` if automatic spectral-axis detection is unsuitable. Raw LAN files are not supported by this converter and must first be converted to one of the accepted formats.

The output is `data/WDC_Mall/real/wdc_mall.npy`, with source information in `data/WDC_Mall/manifest.json`. The evaluation entrypoint applies per-band normalization and a 256 × 256 center crop by default. There is no clean reference, so this dataset supports qualitative inspection, not a ground-truth metric comparison.

## Existing outputs

Chikusei and Washington DC Mall preparation refuse to replace an existing output directory by default. Their `--force` option deletes and recreates that output directory, so use it only when all files there can be discarded.
