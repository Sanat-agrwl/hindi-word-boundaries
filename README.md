# Human Word-Boundary Annotations for Hindi

[![model](https://img.shields.io/badge/model-HuggingFace-yellow)](https://huggingface.co/Sanat-agrwl/indic-whisper-ctc-timestamps)
[![licence](https://img.shields.io/badge/licence-CC%20BY%204.0-blue)](https://creativecommons.org/licenses/by/4.0/)

Manually placed word boundaries for 100 Hindi utterances from the IndicVoices-R test split,
with **71 of them annotated twice by two independent annotators**, so that an
inter-annotator agreement floor can be measured.

Word-timestamp systems for Whisper are normally evaluated by agreement with another aligner,
typically a wav2vec2 CTC model. That is not the same as accuracy, and interpreting such a
number requires knowing how closely two people agree with each other on the same task. This
repository provides that reference point for Hindi, together with the tool used to collect it.

| | |
|---|---|
| **This repository** | the annotations, the agreement statistics, the annotation tool |
| **[Model on Hugging Face](https://huggingface.co/Sanat-agrwl/indic-whisper-ctc-timestamps)** | the Hindi Whisper model with its CTC head, and single-pass inference |

## Inter-annotator agreement

| metric | value |
|---|---|
| boundaries compared | 1858 |
| mean absolute difference | **27.0 ms** |
| median | 20.6 ms |
| p90 | 58.8 ms |
| within 20 ms | 53.3% |
| within 50 ms | 85.3% |

Pooled over word starts and ends after LCS matching of the two word sequences, on the 71
doubly annotated utterances. Reproduce it with `python tool/score_annotations.py`.

## Contents

| file | description |
|---|---|
| `annotations.jsonl` | one record per utterance with all its annotations |
| `annotations.csv` | flattened, one row per annotated word |
| `agreement.json` | the statistics above, machine readable |
| `tool/demo_batch.html` | the annotation interface, runnable as-is |
| `tool/make_annotation_batches.py` | generates batches, including the A/B split |
| `tool/score_annotations.py` | recomputes the agreement figures |
| `model/word_timestamps.py` | audio in, transcript and word timestamps out |
| `model/ctc_head_E.pt` | CTC head for the main model (E) |
| `model/ctc_head_B_vasista22.pt` | CTC head for the public-encoder configuration (B) |

### Record format

```json
{
  "id": "ivr-hi-0114",
  "source": {"corpus": "IndicVoices-R", "language": "Hindi",
             "split": "test", "index": 114},
  "text": "...",
  "duration_s": 1.6,
  "n_annotations": 2,
  "annotations": [
    {"annotator": "ann1", "words": [{"word": "...", "start": 0.05, "end": 0.449}]},
    {"annotator": "ann2", "words": [{"word": "...", "start": 0.04, "end": 0.462}]}
  ]
}
```

Times are seconds from the start of the clip. Words appear in reference order, and every
annotation of an utterance covers the same reference word sequence.

## Annotation protocol

- 100 utterances (11 minutes of speech, 2352 annotated word tokens) drawn from the
  IndicVoices-R Hindi test split.
- Each was placed **twice, in disjoint halves of the batch sequence**, so the two copies were
  annotated by different people. The agreement above is therefore *inter*-annotator, not
  self-agreement.
- Annotators saw a waveform, a spectrogram, a playback cursor with adjustable speed, and the
  reference words. **No model predictions were shown**, so anchoring to a system's output was
  not possible.
- `annotation_key.json`, which maps each presentation back to its source utterance and copy,
  is withheld from annotators: it reveals which presentations are repeats.

## Running the protocol on your own data

```bash
python tool/make_annotation_batches.py --out mybatches --n 100 --batches 10
# annotators open each page and return the JSON from the textarea
python tool/score_annotations.py --dir mybatches
```

Open `tool/demo_batch.html` in a browser first to see the task. Pages are self-contained:
audio and spectrograms are embedded, so a batch can be sent to an annotator who needs no
setup at all. Expect roughly 8 MB per 20 utterances.

## Model

Audio in, transcript and word timestamps out, from **one model in one encoder pass**. The
encoder runs once; the attention decoder reads its output for the transcript and the CTC head
reads the same tensor to place boundaries.

```bash
# main model (E): weights are on the Hub, nothing else needed
python model/word_timestamps.py clip.wav \
    --model Sanat-agrwl/indic-whisper-ctc-timestamps --head model/ctc_head_E.pt

# configuration B: frozen public encoder, no large download
python model/word_timestamps.py clip.wav

python model/word_timestamps.py clip.wav --text "..."        # time a known transcript
python model/word_timestamps.py clip.wav --calibrate         # apply the bias correction
```

**E** is the main model of the paper (14.6% WER, 20.8 ms boundaries); its encoder is on
[Hugging Face](https://huggingface.co/Sanat-agrwl/indic-whisper-ctc-timestamps) because of its
size. **B** places the same kind of head over the frozen public
`vasista22/whisper-hindi-large-v2`, so it runs with nothing else to fetch; it is weaker as a
recogniser (22.9% WER) but its boundaries are equivalent (20.7 ms).

Boundaries from a CTC head are systematically tight: words start about 38 ms late and end
about 100 ms early relative to human annotation. `--calibrate` subtracts those two constants,
which roughly halves the error, from 74.5 to 43.0 ms mean measured out-of-sample. Because the
offsets apply independently, calibrated spans of adjacent words can overlap slightly.

## Licence

The annotations, statistics and tool are released under **CC BY 4.0**, matching IndicVoices-R
from which they derive. Attribution to both this work and the corpus is required; see
`LICENSE`. The model is released separately under Apache-2.0.

The annotation records redistribute no audio. The demo batch under `tool/` embeds a few
seconds of IndicVoices-R audio so that it runs without setup, which CC BY permits with
attribution.

## Citation

If you use these annotations, please cite the accompanying paper (details to follow) and the
IndicVoices-R corpus.
