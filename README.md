# anime_vfr

`anime_vfr` analyzes hybrid MKV sources with IVTCVK, reconstructs a VFR clip,
encodes it, and muxes source tracks.

## Pipeline

1. Anime_VFR opens the source with the selected VapourSynth source filter.
2. A generated VPY passes the source node to `core.ivtcvk.AnalyzeTrack`.
3. IVTCVK writes one editable `*_ivtc.tsv`.
4. TSV decisions and series rules resolve `Unknown` ranges.
5. A reconstruction VPY calls `core.ivtcvk.Reconstruct`.
6. IVTCVK returns the final clip and writes V2 timecodes.
7. Anime_VFR applies geometry, additional VPY processing, encoding, and muxing.

Anime_VFR owns source selection, media metadata, encoding, and muxing. IVTCVK
owns classification, cadence phase, combing analysis, decision resolution,
reconstruction, Bob selection, and decimation.

## Requirements

- Python 3.11 or newer
- VapourSynth API 4
- IVTCVK
- FFMS2, L-SMASH Works, or BestSource
- fmtconv, `vsdeinterlace`, `vsaa`, and QTempGaussMC dependencies
- MKVToolNix, FFmpeg, MediaInfo, and a Y4M-compatible encoder

| Backend | Behavior |
| --- | --- |
| `auto` | Vulkan initialization followed by CPU selection on initialization error |
| `vulkan` | Vulkan processing |
| `cpu` | CPU processing |

## Configuration

Anime_VFR reads `config.toml` from the repository root. `config.example.toml`
contains the supported tool and encoder settings. Tool values accept executable
names or absolute paths. An empty `ivtc_plugin` value selects VapourSynth
autoloading.

## Analyze

```text
python anime_vfr.py SOURCE --analyze-only --work-dir WORK_DIRECTORY
```

The command prints the editable `*_ivtc.tsv` path.

## Encode

```text
python anime_vfr.py SOURCE --output OUTPUT_DIRECTORY --work-dir WORK_DIRECTORY
```

`--reanalyze` replaces the existing TSV and its editable decisions.

## Options

| Option | Meaning |
| --- | --- |
| `--analyze-only` | Generate or reuse the IVTC TSV |
| `--reanalyze` | Replace the IVTC TSV |
| `--backend auto\|vulkan\|cpu` | Select the analysis backend |
| `--analysis-batch N` | Set native analysis batch size |
| `--bob-backend auto\|opencl\|cpu` | Select the NNEDI3 backend used by QTGMC |
| `--source-filter ffms2\|lsmas\|bestsource` | Select the source filter |
| `--field-order tff\|bff` | Set source field order |
| `--ivtc-plugin PATH` | Select the IVTCVK library |
| `--ivtc-config PATH` | Select an Unknown-rule config |
| `--crop L:R:T:B` | Crop after reconstruction |
| `--resize WIDTHxHEIGHT` | Set final dimensions and square-pixel SAR |
| `--yuv444` | Produce 10-bit YUV 4:4:4 output |
| `--additional-vpy PATH` | Apply a VPY snippet after geometry processing |
| `--frames A-B` | Select a half-open output-timeline range |
| `--bob [NUM/DEN]` | Select global CFR Bob processing |
| `--progressive-dedup [N]` | Deduplicate a progressive source |
| `--keep-work` | Retain generated VPY and timing files |
| `--report` | Inspect timestamps of an encoded MKV |

## Artifacts

The classifier artifact is `*_ivtc.tsv`. The work directory also contains
generated VPY scripts and timecode files. Analyze runs and failed runs retain
their work directory. `--keep-work` retains the same files after successful
encoding.

`Reconstruct(timecodes=...)` writes one timestamp for each output frame.
Anime_VFR compares the VSPipe frame count with the timecode count before
encoding.

## Verification

```text
python -m compileall -q .
python anime_vfr.py --help
```

## License

GNU General Public License v3.0. See `LICENSE`.
