# anime_vfr

`anime_vfr` was created for a typical problem found in many NTSC anime DVDs: within the same episode there can be film sections recoverable to 23.976 fps, truly interlaced sections at 59.94 fields/s, digitally overlaid credits or titles, CG, video-rate pans, cadence changes, and non-uniform hard telecine. In these cases the file is often 480i/29.97 at container or MPEG-2 level, but the actual content is not all of the same nature.

The simple case is a regular soft-telecined or hard-telecined DVD: apply IVTC and return to 23.976 fps. The opposite case is a purely interlaced source: apply bob deinterlacing and preserve motion at 59.94 fps. `anime_vfr` is meant for the middle ground, that is, hybrid sources where both decisions are correct only locally. This is a frequent pattern in NTSC anime DVDs from long-running TV series produced during the analog/digital transition period: while working on this script it was verified on `Bleach`, while public technical discussions report similar or nearby problems on `Naruto`, `Naruto Shippuden`, and `One Piece` DVDs. Historical guides also mention hybrid DVDs such as `X TV`, `Azumanga Daioh`, and `Super GALS`, with openings/endings, digital sections, or purely interlaced portions mixed with telecined material.

Global solutions have heavy tradeoffs. Applying IVTC to the whole episode preserves film material well, but on truly interlaced sections it produces judder, incorrect drops, or residual combing. A global bob fixes 60i motion, but also turns film into interpolated 60p, increasing the frame count and degrading detail cleanliness without recovering the original structure. A single CFR conversion to 23.976 or 29.97 always forces part of the material to the wrong timing: either video-rate parts lose fluidity, or film parts get duplicated/interpolated frames.

`anime_vfr` uses a different strategy: classify the content per frame/segment, build a progressive VFR output, and use the correct treatment for each area. Film sections go through IVTC/decimation; truly interlaced sections go through bob deinterlacing; the final mux uses Matroska timecodes to preserve the original timing without forcing everything to one constant framerate.

## Installation

Before running the script, open `config.py` and adapt binary paths to your machine:

- `MKVMERGE`, `MKVEXTRACT`, `VSPIPE`, `PYTHON_BIN`, `MEDIAINFO`, `FFMPEG`
- `ENCODER_BIN`
- `ENCODER_PARAMS`

The included compression parameters are only an example operational preset. You must review and change them according to the encoder you choose, the available GPU/CPU, and the quality level you want. In particular, if you do not use `NVEncC`, you must change both `ENCODER_BIN` and `ENCODER_PARAMS` to a command line compatible with your encoder.

Make sure that `python` is the VapourSynth environment Python, that is, the Python executable able to import `vapoursynth`. Required components: Python packages `numpy`, `vsdeinterlace`, `vsaa`, `vstools`, `vskernels` and their dependencies; VapourSynth plugins BestSource, TIVTC, Vinverse, Sneedif/NNEDI3 OpenCL and the plugins required by `QTempGaussMC`, such as MVTools/RGTools/RemoveGrain or equivalent plugins from your own distribution; external binaries `VSPipe`, `mkvmerge`, `mkvextract`, `MediaInfo`, `ffmpeg`; a video encoder compatible with Y4M input from pipe, for example `ffmpeg`, Rigaya `NVEncC` for NVIDIA NVENC, Rigaya `QSVEncC` for Intel Quick Sync, Rigaya `VCEEncC` for AMD VCE/VCN/AMF, or Rigaya `rkmppenc` for Rockchip MPP. Binary paths and encoder parameters are configured in `config.py`.

Minimal checks:

```powershell
python -m pip install numpy
python -c "import vapoursynth as vs; import numpy; print(vs.__version__)"
python -c "import vapoursynth as vs; c=vs.core; print(hasattr(c,'bs'), hasattr(c,'tivtc'), hasattr(c,'vinverse'), hasattr(c,'sneedif'))"
python -c "from vsdeinterlace.qtgmc import QTempGaussMC; from vsaa import NNEDI3; print(NNEDI3(opencl=True)._deinterlacer_function)"
```

In the code, the bob branch uses:

```python
QTempGaussMC(
    clip,
    basic_bobber=NNEDI3(nsize=4, nns=4, qual=2, opencl=True),
    tff=True,
    basic_tr=3,
    final_tr=2,
    source_match_mode=QTempGaussMC.SourceMatchMode.TWICE_REFINED,
).deinterlace()
```

With `vsaa.NNEDI3(opencl=True)`, the wrapper selects the OpenCL backend `core.lazy.sneedif.NNEDI3`; with `opencl=False`, it would use the CPU backend `core.lazy.znedi3.nnedi3` instead.

## Quick Start

```powershell
cd C:\Users\Simone\anime_vfr
python anime_vfr.py "C:\video\episode.mkv" --analyze-only
python anime_vfr.py "C:\video\episode.mkv" --output "D:\encoded"
```

## Parameters

Syntax:

```text
anime_vfr.py source [options]
```

| Parameter               | Short description                                                                                   |
| ----------------------- | --------------------------------------------------------------------------------------------------- |
| `source`                | Source MKV file, or folder containing MKV files.                                                     |
| `--report`              | Analyzes already produced MKV files by reading timestamps_v2; does not run the pipeline.             |
| `--analyze-only`        | Runs analysis, classification, dedup, timecodes and VPY generation, but skips encode/mux.            |
| `--bob`                 | Forces the whole title to 60p bob and skips TIVTC/classifier/dedup.                                  |
| `--progressive-dedup`   | Deduplicates an already progressive source without TIVTC/classifier/bob.                             |
| `--dedup`               | Enables dedup on film segments in the hybrid pipeline.                                               |
| `--output PATH`         | Output folder; when specified, keeps the source filename.                                            |
| `--work-dir PATH`       | Working folder; if omitted, uses `<output>\work`.                                                    |
| `--keep-work`           | Keeps intermediate files instead of cleaning the work dir at the end of processing.                  |
| `--additional-vpy PATH` | Appends a VPY snippet to the final pass; it is not a standalone VPY script.                          |
| `--frames RANGE`        | Processes only an output frame range, in `N` or `A-B` format.                                        |
| `--strip-audio`         | Does not mux source audio tracks.                                                                    |
| `--strip-sub`           | Does not mux source subtitle tracks.                                                                 |

### `source`

In normal mode this is a single MKV or a folder of MKV files to process. With `--report` it can be an already produced MKV, or a folder containing already produced MKV files.

```powershell
python anime_vfr.py "C:\video\episode.mkv"
python anime_vfr.py "D:\encoded" --report
```

### `--report`

Does not use the classifier and does not rebuild the pipeline. It extracts timestamps_v2 from the final MKV, measures frame intervals, and prints an after-the-fact distribution in the classes `<24`, `24`, `24<x<60`, `60`, `>60`. It is meant to check the final result, not to decide how to encode.

```powershell
python anime_vfr.py "D:\encoded\episode.mkv" --report
python anime_vfr.py "D:\encoded" --report
```

### `--analyze-only`

Runs the pipeline up to final timecode generation and VPY script generation, then stops before encode and mux. It prints to console the 24/60 classification, dedup statistics, a pre-dedup FPS histogram over 20 buckets, and a dedup-drop histogram over 20 buckets.

```powershell
python anime_vfr.py "C:\video\episode.mkv" --analyze-only
python anime_vfr.py "C:\video\episode.mkv" --analyze-only > analysis.txt
```

### `--bob`

Forces the whole title to 60p bob. It is useful when you already know the source is truly interlaced and you do not want to attempt film recovery. In this mode TFM/TDecimate, the classifier, and dedup are not executed.

```powershell
python anime_vfr.py "C:\video\episode.mkv" --bob
```

### `--progressive-dedup`

Uses only the dedup/timecode part on an already progressive source. It skips TIVTC, the classifier, and bob: every source frame is treated as a valid progressive frame, then visually duplicated frames are removed and their duration is transferred to VFR timecodes.

This mode is meant for sources that do not need IVTC or deinterlacing, but contain real holds/duplicates that should be compacted into VFR. It is not suitable for interlaced sources or telecined sources that have not already been resolved.

```powershell
python anime_vfr.py "C:\video\progressive.mkv" --progressive-dedup --analyze-only
python anime_vfr.py "C:\video\progressive.mkv" --progressive-dedup --output "D:\encoded"
```

### `--dedup`

Enables dedup on film segments in the hybrid pipeline. Without this flag, the pipeline performs IVTC, 24/60 classification, bob of truly interlaced sections, and VFR timecode generation, but keeps all decimated film frames. `--dedup` is used when you want to compact holds and visual duplicates by transferring their duration to VFR timecodes.

```powershell
python anime_vfr.py "C:\video\episode.mkv" --dedup
```

### `--output` and `--work-dir`

`--output` chooses where final MKV files are written. If it is specified, the final file keeps exactly the source filename. If it is omitted, output is written in the same folder as the source with suffix `_1`, so the original file is not overwritten.

`--work-dir` chooses where processing artifacts are created. Each processing run uses a unique subfolder, so multiple instances can share the same work dir without overwriting intermediate files. Without `--keep-work`, the run subfolder is cleaned at the end; user-requested outputs do not depend on `--keep-work`.

```powershell
python anime_vfr.py "C:\video\episode.mkv" --output "D:\encoded"
python anime_vfr.py "C:\video\episode.mkv" --work-dir "E:\temp\anime_vfr"
```

Output examples:

```text
python anime_vfr.py "C:\video\episode.mkv"
  -> C:\video\episode_1.mkv

python anime_vfr.py "C:\video\episode.mkv" --output "D:\encoded"
  -> D:\encoded\episode.mkv
```

### `--additional-vpy`

Appends a VPY snippet to the final pass generated by `anime_vfr`. It is not a standalone VPY: it is copied into pass2b after the pipeline has already created the final VFR clip.

When the snippet runs, these already exist:

- `import vapoursynth as vs`
- `core = vs.core`
- `clip`, that is, the clip already assembled from 24p/60p segments, already resized to square pixels and converted to `YUV420P10`

The snippet operates on the `clip` variable and reassigns it. A typical case is clearing a residual interlaced flag and forcing a technical CFR tag before filters:

```python
clip = core.std.SetFrameProp(clip, prop="_FieldBased", intval=0)
clip = core.std.AssumeFPS(clip, fpsnum=30000, fpsden=1001)

from vstools import depth
clip = depth(clip, 16)

from vsdeband import placebo_deband
clip = placebo_deband(clip, radius=8.0, thr=3.0, iterations=4, grain=0.0)
```

Do not reload the source with `VideoSource`, and do not call `clip.set_output(0)`: `anime_vfr` adds it at the end. The snippet must preserve frame count and frame order, because the VFR timecodes have already been generated.

Usage example:

```powershell
python anime_vfr.py "C:\video\episode.mkv" --additional-vpy "C:\filters\filters.vpy"
```

The same model is used by snippets generated by tools such as `gto_crop_detect.py`: the produced file operates directly on `clip` and is appended to pass2b.

### `--frames`

Limits the encode to an output frame range. With a single number `N`, it processes `0-N`; with `A-B`, it processes the indicated interval. When active, audio and subtitles are trimmed with FFmpeg before muxing.

```powershell
python anime_vfr.py "C:\video\episode.mkv" --frames 1500
python anime_vfr.py "C:\video\episode.mkv" --frames 100-5000
```

### `--strip-audio`, `--strip-sub`

`--strip-audio` and `--strip-sub` respectively exclude audio and subtitles from the final mux.

```powershell
python anime_vfr.py "C:\video\episode.mkv" --strip-audio
python anime_vfr.py "C:\video\episode.mkv" --strip-sub
```

## How It Works

The pipeline does not decide the final framerate by looking only at TDecimate output. TIVTC is used to build a coherent film timeline and to extract useful signals, but the 24p/60p decision is made on source frames with a multi-metric classifier. This is necessary on mixed anime sources: almost static scenes, incomplete patterns, or isolated combed frames can mislead a single indicator. The pipeline cross-checks multiple signals and eventually reduces everything to a binary choice: film or bob.

First, metadata, SAR, and source timestamps_v2 are read. Source timestamps are the time base: they are used to generate final timecodes and to close the last segment correctly as well, avoiding cumulative drift against audio. The resolution is converted to square pixels while preserving display aspect ratio.

The initial TIVTC pass produces field-match and decimation information. TFM searches for matches between fields to recover progressive frames from telecined material; TDecimate builds the decimated timeline. During the second pass, a framemap is created linking every frame in the decimated timeline to the source frame it comes from. This link is essential: classification works on the source, but film assembly works on the decimated timeline.

The classifier considers each source frame as the center of a ten-field window. In that window it measures which fields are almost equal to the previous field using a 16x16 luma difference: below threshold, they are considered matching. 3:2 telecine produces a recognizable cyclic structure; to avoid false positives, the code accepts telecine phase only in the strict case with exactly two matches five fields apart. If this pattern is present and the frame is not combed, the frame is classified as 24p. If it is combed, the TFM flag prevails and the frame is treated as true 60i.

The pattern alone is not enough. The pipeline also measures field motion and an FFT ratio on vertical Nyquist energy from the luma plane. Nyquist energy catches alternating-line structure, typical of interlacing with motion; very low values, combined with enough motion, can promote progressive/telecine material even when the cyclic pattern is incomplete. Very static scenes are handled cautiously: if almost all fields match and motion is minimal, the decision is not treated as strong evidence of 60i, because static progressive material can also have nearly identical fields.

After initial classification, consistency passes are applied. Isolated telecine anchors are rejected if they do not have enough coherent neighbors; small telecine clusters surrounded by 60i are reclassified; ambiguous frames inherit from local density and from the nearest anchor, with a bias consistent with why they were ambiguous. Finally, on sufficiently long 60i clusters, speculative IVTC verification is performed: slow TFM is applied on a subclip and the number of frames that remain combed is checked. A cluster is recovered as telecine only if IVTC cleans it and if average motion is sufficient; this avoids confusing static scenes with true 24p recovery.

At the end, the pipeline normalizes everything into two operational classes. Only frames classified as `interlaced_60i` become `video_bob`; everything else enters the `film` branch. The framemap is rewritten with this binary choice and grouped into contiguous segments. In bob segments, exact source indices are preserved, because TDecimate may have skipped source frames inside the same visual area and the pipeline must not reinsert wrong frames.

Dedup works only on film segments. It reconstructs the same decimated stream that will be used during encode, compares consecutive film frames, and groups duplicate runs up to the configured limit. If it finds, for example, a 4-in-1 run, it keeps a single video frame but extends timing through timecodes. This way the visual content is not repeated unnecessarily, while duration remains locked to the source.

Final timecodes are generated from the source timeline and segmentation. In film segments, one timestamp is written for each kept decimated frame; with dedup, the next timestamp advances by the duration represented by the run. In bob segments, each source frame generates two output frames, so the source duration is split in two. The last timestamp is closed using duration estimated from source timestamps, not a fixed theoretical framerate.

The final VPY builds only the required branches. If there are film segments, it creates the TFM/TDecimate branch and applies Vinverse on residual combed frames. If there are 60p segments, it creates the QTempGaussMC branch with NNEDI3 OpenCL. Segments are then assembled with `core.std.Splice`. `AssumeFPS` in the VPY is only a technical stream tag; the real timing in the final MKV is determined by the muxed Matroska timecodes.

In normal mode, `VSPipe` sends Y4M to the configured encoder, then `mkvmerge` muxes video, VFR timecodes, audio, and source subtitles. In `--analyze-only`, the pipeline stops before encode but still produces the technical report on standard output. In `--report`, however, no decision is rerun: it only measures the final already produced file.

## Code Structure

```text
anime_vfr.py   CLI entrypoint
pipeline.py    orchestration, TIVTC passes, classifier and final VPY
config.py      binary paths and pipeline parameters
media.py       video metadata and source timestamps
segments.py    framemap and film/bob segmentation
dedup.py       dedup on film segments
timecodes.py   final VFR timecode generation
report.py      analyze-only and after-the-fact reports
encode.py      encode and final mux
utils.py       shared helpers
```

## Operational Notes

- The final model is always binary: 24p film or 60p bob.
- `--analyze-only` is the correct way to validate a new source before encoding.
- `--report` is only for already produced files: it measures MKV timestamps, it does not repeat classification.
- Dedup is applied only to film segments and changes the video frame count, not the temporal duration.
- `--progressive-dedup` applies the same principle to already progressive sources, without 24/60 classification.
- `--additional-vpy` scripts must preserve frame count and frame order.
- If you change encoder, update both `ENCODER_BIN` and `ENCODER_PARAMS` in `config.py`.

## License

This project is distributed under the GNU General Public License v3.0. See `LICENSE`.

## Buy me a coffee!

[!["Buy Me A Coffee"](https://www.buymeacoffee.com/assets/img/custom_images/orange_img.png)](https://www.buymeacoffee.com/simonefil)
