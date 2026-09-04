"""Transcript and word timestamps from one model in one encoder pass.

Audio goes through the Whisper encoder once. The resulting frame representations are used
twice: the attention decoder reads them to produce the transcript, and a character CTC head
reads the same tensor to place word boundaries by forced alignment at 20 ms resolution. No
second model and no second forward pass through the encoder are involved, which is the point
of the method.

    python word_timestamps.py clip.wav                       # transcribe and time
    python word_timestamps.py clip.wav --text "..."          # time a known transcript
    python word_timestamps.py clip.wav --calibrate           # apply the measured bias

Boundaries from a CTC head are systematically tight: words start about 38 ms late and end
about 100 ms early relative to human annotation. Subtracting those two constants roughly
halves the error, from 74.5 to 43.0 ms mean measured out-of-sample. --calibrate applies them.
Because the two offsets are applied independently, calibrated spans of adjacent words can
overlap by a few tens of milliseconds; clamp them if you need disjoint spans.
"""

import argparse, json, unicodedata
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import soundfile as sf
from torchaudio.functional import forced_align
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

HERE = Path(__file__).parent
SR, HOP = 16000, 320                              # 20 ms frames
ONSET_BIAS_S, OFFSET_BIAS_S = 0.0389, -0.1030     # measured against human annotation


def load_audio(path):
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1: audio = audio.mean(1)
    if sr != SR:
        from math import gcd
        import scipy.signal as ss
        g = gcd(SR, sr)
        audio = ss.resample_poly(audio, SR//g, sr//g).astype(np.float32)
    return audio


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("audio")
    ap.add_argument("--model", default="vasista22/whisper-hindi-large-v2",
                    help="ASR model supplying the encoder and decoder. Must match the head.")
    ap.add_argument("--head", default=str(HERE/"ctc_head_B_vasista22.pt"))
    ap.add_argument("--vocab", default=str(HERE/"char_vocab.json"))
    ap.add_argument("--text", default=None,
                    help="align this transcript instead of the model's own output")
    ap.add_argument("--calibrate", action="store_true")
    a = ap.parse_args()

    vocab = json.load(open(a.vocab, encoding="utf-8"))
    blank, space = vocab["<blank>"], vocab["<space>"]

    def ids(t):
        o = []
        for ch in unicodedata.normalize("NFC", t):
            if ch.isspace():
                if o and o[-1] != space: o.append(space)
            elif ch in vocab: o.append(vocab[ch])
        return o

    audio = load_audio(a.audio)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dt = torch.float16 if dev == "cuda" else torch.float32

    proc = AutoProcessor.from_pretrained(a.model)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        a.model, torch_dtype=dt, low_cpu_mem_usage=True).to(dev).eval()
    ck = torch.load(a.head, map_location="cpu")
    head = nn.Linear(model.config.d_model, ck["vocab_size"])
    head.load_state_dict(ck["state_dict"])
    head = head.to(dev, dtype=dt).eval()

    # Whisper ships two flavours of generation config. forced_decoder_ids is silently
    # ignored by the newer one, which leaves the model translating into English.
    gc = model.generation_config
    if getattr(gc, "lang_to_id", None) is not None:
        gc.forced_decoder_ids = None
        gen_kw = {"language": "hi", "task": "transcribe"}
    else:
        gc.forced_decoder_ids = proc.get_decoder_prompt_ids(language="hi",
                                                            task="transcribe")
        gen_kw = {}

    feats = proc.feature_extractor(audio, sampling_rate=SR,
                                   return_tensors="pt").input_features.to(dev, dtype=dt)

    with torch.no_grad():
        # ---- the single encoder pass -------------------------------------------------
        enc_out = model.model.encoder(feats)

        if a.text is None:
            # the decoder reads the encoder output that already exists rather than
            # re-encoding, so the transcript costs no additional encoder work
            gen = model.generate(encoder_outputs=enc_out, **gen_kw)
            text = proc.batch_decode(gen, skip_special_tokens=True)[0].strip()
        else:
            text = a.text.strip()

        # the CTC head reads the same tensor
        log_probs = head(enc_out.last_hidden_state).float().log_softmax(-1)

    words, lab = text.split(), ids(text)
    if not words or not lab:
        raise SystemExit(f"no alignable Devanagari text (transcript was {text!r})")

    # the true frame count for this clip, not the padded 1500-frame window: passing the
    # padded length makes blank near-optimal on short clips and destroys the alignment
    nf = min(1500, max(len(lab), int(np.ceil(len(audio)/HOP))))
    tok, _ = forced_align(log_probs[:, :nf, :], torch.tensor([lab], device=dev),
                          torch.tensor([nf], device=dev),
                          torch.tensor([len(lab)], device=dev), blank=blank)

    pos2frames, pos, prev = {}, -1, -1
    for t, tk in enumerate(tok[0].tolist()):
        if tk == blank: prev = tk; continue
        if tk != prev: pos += 1
        prev = tk; pos2frames.setdefault(pos, []).append(t)

    spans, p = [], 0
    for w in words:
        i = ids(w)
        spans.append((p, p+len(i)-1) if i else None)
        if i: p += len(i)+1

    print(f"# {text}")
    for w, sp in zip(words, spans):
        if sp is None: continue
        fr = [x for c in range(sp[0], sp[1]+1) for x in pos2frames.get(c, [])]
        if not fr: continue
        st, en = min(fr)*HOP/SR, max(fr)*HOP/SR
        if a.calibrate:
            st, en = st - ONSET_BIAS_S, en - OFFSET_BIAS_S
        print(f"{st:8.3f}  {en:8.3f}  {w}")


if __name__ == "__main__":
    main()
