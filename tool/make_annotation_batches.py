"""
Build self-contained HTML pages for human word-boundary annotation.

Design constraints, in order of importance:

  1. NO MODEL PREDICTIONS ARE EMBEDDED. Not ours, not wav2vec2's. Showing a boundary makes
     the annotator nudge it rather than judge it, and the result would measure agreement
     with ourselves. The timeline starts empty. This is the whole reason the exercise is
     worth doing, so it is enforced here rather than left to discipline.

  2. REPEATS FOR A NOISE FLOOR. A fraction of utterances appear twice (in different
     batches, ids disguised) so inter/intra-annotator agreement can be measured. Without
     that number, "our model is within 21 ms" has nothing to be compared against: if humans
     disagree with each other by 30 ms then 21 ms is at the noise floor and the paper can
     say so.

  3. THE ANNOTATOR NEVER TYPES TEXT. The word sequence is known, so the task is placing
     boundaries on a timeline -- minutes per utterance, not hours.

Each page embeds audio as a WAV data URI and a precomputed spectrogram as a PNG data URI
(computing an FFT in the browser buys nothing and costs latency). Output is JSON, either
downloaded directly (local file) or copied from a textarea (published artifact, whose
sandbox blocks downloads).

    python scripts/22_make_annotation_batches.py --n 60 --batch-size 25 --repeat-frac 0.15
"""

import argparse, base64, io, json, random, wave
from pathlib import Path

import numpy as np

BASE = Path(__file__).parent.parent
SR = 16000


def wav_data_uri(audio, sr=SR):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1)*32767).astype("<i2").tobytes())
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode()


def spectrogram_png(audio, sr=SR, height=180, nfft=512, hop=80):
    """Log-magnitude spectrogram, 0-8 kHz, as a PNG data URI.

    Precomputed rather than done in JS: formant and burst structure is where word
    boundaries actually are, and a browser FFT would only add latency.
    """
    try:
        from PIL import Image
    except ImportError:
        return ""
    n = 1 + max(0, (len(audio)-nfft)//hop)
    if n < 2: return ""
    win = np.hanning(nfft)
    S = np.empty((nfft//2, n), dtype=np.float32)
    for i in range(n):
        seg = audio[i*hop:i*hop+nfft]
        if len(seg) < nfft: seg = np.pad(seg, (0, nfft-len(seg)))
        S[:, i] = np.abs(np.fft.rfft(seg*win)[:nfft//2])
    S = 20*np.log10(S + 1e-8)
    S = np.clip((S - S.max() + 70)/70, 0, 1)          # 70 dB dynamic range
    img = (255*(1-S)).astype(np.uint8)[::-1]          # low freq at bottom, dark = energy
    im = Image.fromarray(img).resize((max(n, 400), height))
    buf = io.BytesIO(); im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def build_items(aligned, n_samples, n_batches, seed):
    """Every sample twice, with the two copies in DIFFERENT halves of the batch set.

    Copies land in batches 1..H and H+1..N respectively, so annotator 1 takes the first
    half and annotator 2 the second and every utterance receives two INDEPENDENT
    annotations. Repeats inside one person's workload would only measure self-consistency;
    across people it measures the real human noise floor, which is the number that decides
    whether a model at 21 ms is good.

    The two copies carry unrelated ids so neither annotator can tell they are paired.
    """
    rng = random.Random(seed)
    rows = []
    for i, line in enumerate(open(aligned, encoding="utf-8")):
        if not line.strip(): continue
        s_ = json.loads(line)
        text = (s_.get("verbatim_transcript") or s_.get("verbatim") or "").strip()
        audio = np.asarray(s_["audio_array"], dtype=np.float32)
        if not text or audio.size == 0: continue
        rows.append({"idx": i, "text": text, "audio": audio})
    rng.shuffle(rows)
    rows = rows[:n_samples]

    half = n_batches // 2
    per = max(1, len(rows) // max(half, 1))
    buckets = [[] for _ in range(n_batches)]

    orderA = list(range(len(rows))); rng.shuffle(orderA)
    orderB = list(range(len(rows))); rng.shuffle(orderB)
    for pos, ri in enumerate(orderA):
        buckets[min(pos // per, half - 1)].append((rows[ri], "A"))
    for pos, ri in enumerate(orderB):
        buckets[half + min(pos // per, n_batches - half - 1)].append((rows[ri], "B"))

    out, n = [], 0
    for b, bucket in enumerate(buckets):
        rng.shuffle(bucket)
        for r, copy in bucket:
            n += 1
            out.append(dict(r, uid=f"s{n:04d}", copy=copy, batch=b + 1))
    return out


def render(batch, batch_no, n_batches):
    payload = []
    for it in batch:
        uid = it["uid"]
        payload.append({
            "uid": uid,
            "words": it["text"].split(),
            "duration": round(len(it["audio"])/SR, 4),
            "wav": wav_data_uri(it["audio"]),
            "spec": spectrogram_png(it["audio"]),
        })
    data = json.dumps(payload, ensure_ascii=False)
    return HTML.replace("__DATA__", data)\
               .replace("__BATCH__", str(batch_no))\
               .replace("__NBATCH__", str(n_batches))


HTML = r"""<meta charset="utf-8">
<title>Word Boundary Annotation</title>
<style>
  :root{
    --bg:#faf9f7; --panel:#fff; --ink:#1a1a1a; --muted:#6b6b6b; --line:#e2e0dc;
    --accent:#b4522d; --accent-soft:#f3e2da; --ok:#2d6a4f; --grid:#cfcbc4;
  }
  @media (prefers-color-scheme:dark){:root:not([data-theme=light]){
    --bg:#16151a; --panel:#1f1e24; --ink:#eceaf0; --muted:#9d99a6; --line:#32303a;
    --accent:#e0794f; --accent-soft:#3a2a22; --ok:#6cc39a; --grid:#3a3743;}}
  :root[data-theme=dark]{--bg:#16151a; --panel:#1f1e24; --ink:#eceaf0; --muted:#9d99a6;
    --line:#32303a; --accent:#e0794f; --accent-soft:#3a2a22; --ok:#6cc39a; --grid:#3a3743;}
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:15px/1.5 ui-sans-serif,system-ui,"Segoe UI",Roboto,sans-serif}
  .wrap{max-width:1080px;margin:0 auto;padding:18px 20px 60px}
  header{display:flex;align-items:baseline;gap:14px;flex-wrap:wrap;margin-bottom:14px}
  h1{font-size:17px;margin:0;letter-spacing:-.01em}
  .sub{color:var(--muted);font-size:13px}
  .bar{height:6px;background:var(--line);border-radius:3px;overflow:hidden;flex:1;min-width:140px}
  .bar>i{display:block;height:100%;background:var(--accent);width:0}
  .panel{background:var(--panel);border:1px solid var(--line);border-radius:10px;
         padding:14px;margin-bottom:14px}
  .canvasWrap{position:relative;cursor:crosshair;user-select:none}
  canvas{display:block;width:100%;border-radius:6px}
  #spec{height:150px;object-fit:fill;width:100%;border-radius:6px 6px 0 0;display:block}
  .words{display:flex;flex-wrap:wrap;gap:6px;margin-top:12px}
  .w{padding:4px 9px;border:1px solid var(--line);border-radius:6px;font-size:15px;
     color:var(--muted);white-space:nowrap}
  .w.done{color:var(--ink);border-color:var(--ok);background:color-mix(in srgb,var(--ok) 10%,transparent)}
  .w.cur{color:#fff;background:var(--accent);border-color:var(--accent);font-weight:600}
  .keys{display:flex;flex-wrap:wrap;gap:8px 18px;color:var(--muted);font-size:12.5px}
  .keys b{color:var(--ink);font-weight:600;font-family:ui-monospace,monospace}
  button{font:inherit;padding:7px 13px;border:1px solid var(--line);background:var(--panel);
         color:var(--ink);border-radius:7px;cursor:pointer}
  button:hover{border-color:var(--accent)}
  button.primary{background:var(--accent);color:#fff;border-color:var(--accent)}
  .row{display:flex;gap:9px;align-items:center;flex-wrap:wrap}
  textarea{width:100%;height:150px;font:12px/1.45 ui-monospace,monospace;padding:10px;
           border:1px solid var(--line);border-radius:8px;background:var(--bg);color:var(--ink)}
  .hint{color:var(--muted);font-size:12.5px;margin-top:8px}
  .transport{display:flex;align-items:center;gap:11px;margin-top:10px}
  #seek{flex:1;-webkit-appearance:none;appearance:none;height:22px;background:transparent;cursor:pointer}
  #seek::-webkit-slider-runnable-track{height:6px;border-radius:3px;background:var(--line)}
  #seek::-moz-range-track{height:6px;border-radius:3px;background:var(--line)}
  #seek::-webkit-slider-thumb{-webkit-appearance:none;width:15px;height:15px;margin-top:-4.5px;
    border-radius:50%;background:var(--accent);border:2px solid var(--panel);cursor:grab}
  #seek::-moz-range-thumb{width:15px;height:15px;border-radius:50%;background:var(--accent);
    border:2px solid var(--panel);cursor:grab}
  #time{font:12.5px/1 ui-monospace,monospace;color:var(--muted);min-width:96px;text-align:right}
  #rate{font:inherit;padding:4px 6px;border:1px solid var(--line);border-radius:6px;
        background:var(--panel);color:var(--ink);cursor:pointer}
  .tag{font-family:ui-monospace,monospace;font-size:12px;color:var(--muted)}
</style>

<div class="wrap">
<header>
  <h1>Word boundary annotation</h1>
  <span class="sub">batch __BATCH__ / __NBATCH__</span>
  <span class="tag" id="uid"></span>
  <div class="bar"><i id="prog"></i></div>
  <span class="sub" id="count"></span>
</header>

<div class="panel">
  <div class="canvasWrap" id="cw">
    <img id="spec" alt="">
    <canvas id="wave" height="120"></canvas>
  </div>
  <div class="words" id="words"></div>
</div>

<div class="panel">
  <div class="transport">
    <button id="playBtn" class="primary" style="min-width:44px">▶</button>
    <input type="range" id="seek" min="0" max="1000" value="0" step="1">
    <span id="time">0.00 / 0.00</span>
    <select id="rate" title="playback speed">
      <option value="0.25">0.25x</option>
      <option value="0.5">0.5x</option>
      <option value="0.75">0.75x</option>
      <option value="1" selected>1x</option>
    </select>
  </div>
  <div class="row" style="margin-top:10px">
    <button id="play" class="primary">▶ play (space)</button>
    <button id="playw">▶ play word (shift+space)</button>
    <button id="loop">loop: off (L)</button>
    <button id="prev">← previous word</button>
    <button id="next">next word (enter) →</button>
    <button id="prevUtt">◀ previous utterance</button>
    <button id="skip">next utterance ▶</button>
  </div>
  <div class="keys" style="margin-top:11px">
    <span><b>click</b> set start</span>
    <span><b>shift+click</b> set end</span>
    <span><b>←/→</b> nudge start 10 ms</span>
    <span><b>,/.</b> nudge end 10 ms</span>
    <span><b>backspace</b> previous</span>
    <span><b>u</b> undo word</span>
    <span><b>drag slider</b> scrub / replay any part</span>
    <span><b>[ / ]</b> previous / next utterance</span>
  </div>
  <div class="hint">Mark where you <em>hear</em> each word begin and end. If two words run
    together, the boundary is one point — set the start of the next word and the previous
    word's end follows automatically. Leave a gap only where you hear a real pause.</div>
</div>

<div class="panel">
  <div class="row">
    <button id="save" class="primary">finish &amp; show JSON</button>
    <button id="dl">download .json</button>
    <span class="sub" id="status"></span>
  </div>
  <textarea id="out" placeholder="Completed annotations appear here — select all and copy."></textarea>
  <div class="hint">Work is saved in this browser as you go, so a refresh will not lose it.</div>
</div>
</div>

<script>
const DATA = __DATA__;
const KEY = "wba_batch___BATCH__";
let ai = 0, wi = 0, ann = {}, loop = false, playTimer = null;
const audio = new Audio();
const wave = document.getElementById('wave'), ctx = wave.getContext('2d');
let peaks = null;

try { ann = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch(e){ ann = {}; }

function cur(){ return DATA[ai]; }
function marks(){ const u = cur().uid; if(!ann[u]) ann[u] = {}; return ann[u]; }
function persist(){ try{ localStorage.setItem(KEY, JSON.stringify(ann)); }catch(e){} }

function decodePeaks(){
  // draw the waveform from the decoded PCM in the data URI (no extra fetch)
  const b64 = cur().wav.split(',')[1];
  const bin = atob(b64);
  const bytes = new Uint8Array(bin.length);
  for(let i=0;i<bin.length;i++) bytes[i]=bin.charCodeAt(i);
  const pcm = new Int16Array(bytes.buffer, 44, Math.floor((bytes.length-44)/2));
  const W = 1200, out = new Float32Array(W);
  const step = Math.max(1, Math.floor(pcm.length/W));
  for(let i=0;i<W;i++){
    let m=0; const s=i*step;
    for(let j=s;j<Math.min(s+step,pcm.length);j++) m=Math.max(m,Math.abs(pcm[j]));
    out[i]=m/32768;
  }
  peaks = out;
}

function draw(){
  const W = wave.clientWidth, H = wave.height;
  wave.width = W;
  const css = getComputedStyle(document.documentElement);
  ctx.clearRect(0,0,W,H);
  ctx.fillStyle = css.getPropertyValue('--panel'); ctx.fillRect(0,0,W,H);
  // waveform
  ctx.strokeStyle = css.getPropertyValue('--grid'); ctx.lineWidth = 1;
  ctx.beginPath();
  for(let x=0;x<W;x++){
    const v = peaks[Math.floor(x/W*peaks.length)] || 0;
    ctx.moveTo(x+.5, H/2 - v*H*0.46); ctx.lineTo(x+.5, H/2 + v*H*0.46);
  }
  ctx.stroke();
  const dur = cur().duration, m = marks();
  // committed spans
  for(const k in m){
    if(m[k].s==null) continue;
    const x0 = m[k].s/dur*W, x1 = (m[k].e==null? m[k].s : m[k].e)/dur*W;
    ctx.fillStyle = (+k===wi) ? css.getPropertyValue('--accent-soft') : 'rgba(120,120,120,.16)';
    ctx.fillRect(x0, 0, Math.max(1.5, x1-x0), H);
    ctx.strokeStyle = (+k===wi) ? css.getPropertyValue('--accent') : css.getPropertyValue('--grid');
    ctx.beginPath(); ctx.moveTo(x0,0); ctx.lineTo(x0,H);
    ctx.moveTo(x1,0); ctx.lineTo(x1,H); ctx.stroke();
  }
  // playhead
  if(!audio.paused){
    const x = audio.currentTime/dur*W;
    ctx.strokeStyle = css.getPropertyValue('--ok'); ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke();
  }
}

function renderWords(){
  const box = document.getElementById('words'); box.innerHTML='';
  const m = marks();
  cur().words.forEach((w,i)=>{
    const d = document.createElement('span');
    d.className = 'w' + (i===wi?' cur':'') + (m[i]&&m[i].s!=null&&m[i].e!=null?' done':'');
    d.textContent = w;
    d.onclick = ()=>{ wi=i; renderWords(); draw(); };
    box.appendChild(d);
  });
  document.getElementById('uid').textContent = cur().uid;
  document.getElementById('count').textContent =
    `utterance ${ai+1}/${DATA.length} · word ${wi+1}/${cur().words.length}`;
  const done = DATA.filter(d=>{
    const a=ann[d.uid]; if(!a) return false;
    return d.words.every((_,i)=>a[i]&&a[i].s!=null&&a[i].e!=null);
  }).length;
  document.getElementById('prog').style.width = (100*done/DATA.length)+'%';
}

function load(){
  const d = cur();
  audio.src = d.wav;
  document.getElementById('spec').src = d.spec || '';
  decodePeaks(); wi = 0; renderWords(); draw();
  audio.pause(); audio.currentTime = 0; seek.value = 0;
  audio.playbackRate = parseFloat(rateSel.value);   // survives the src change
  syncTransport();
}

function setMark(t, isEnd){
  const m = marks();
  if(!m[wi]) m[wi] = {s:null,e:null};
  if(isEnd) m[wi].e = t; else {
    m[wi].s = t;
    if(m[wi].e==null || m[wi].e < t) m[wi].e = Math.min(cur().duration, t+0.25);
    if(wi>0 && m[wi-1] && m[wi-1].e==null) m[wi-1].e = t;   // connected speech
  }
  persist(); renderWords(); draw();
}

document.getElementById('cw').addEventListener('click', e=>{
  const r = e.currentTarget.getBoundingClientRect();
  const t = Math.max(0, Math.min(cur().duration, (e.clientX-r.left)/r.width*cur().duration));
  setMark(+t.toFixed(3), e.shiftKey);
});

function playRange(a,b){
  clearTimeout(playTimer);
  audio.currentTime = a; audio.play();
  const tick = ()=>{ draw(); syncTransport(); if(!audio.paused) requestAnimationFrame(tick); };
  requestAnimationFrame(tick);
  if(b!=null){
    playTimer = setTimeout(()=>{
      if(loop) playRange(a,b); else { audio.pause(); draw(); }
    }, Math.max(60,(b-a)*1000));
  }
}
function playWord(){
  const m = marks()[wi];
  if(m && m.s!=null) playRange(Math.max(0,m.s-0.12), (m.e!=null?m.e:m.s)+0.12);
  else playRange(0,null);
}

// ---- transport: scrub anywhere, replay without waiting for the clip to finish ----
const seek = document.getElementById('seek'), timeEl = document.getElementById('time');
let scrubbing = false;
function fmt(t){ return (t||0).toFixed(2); }
function syncTransport(){
  const d = cur().duration || 1;
  if(!scrubbing) seek.value = Math.round(1000*(audio.currentTime||0)/d);
  timeEl.textContent = fmt(audio.currentTime) + ' / ' + fmt(d);
  document.getElementById('playBtn').textContent = audio.paused ? '▶' : '❚❚';
}
seek.addEventListener('input', ()=>{
  scrubbing = true;
  const t = seek.value/1000*(cur().duration||1);
  audio.currentTime = t; timeEl.textContent = fmt(t)+' / '+fmt(cur().duration); draw();
});
seek.addEventListener('change', ()=>{ scrubbing = false; });
audio.addEventListener('timeupdate', syncTransport);
audio.addEventListener('play', syncTransport);
audio.addEventListener('pause', syncTransport);
audio.addEventListener('ended', ()=>{ syncTransport(); draw(); });
document.getElementById('playBtn').onclick = ()=>{
  clearTimeout(playTimer);                 // cancel any word-range stop
  if(audio.paused) playRange(audio.currentTime||0, null); else audio.pause();
};
document.getElementById('play').onclick = ()=>{
  clearTimeout(playTimer);
  audio.paused? playRange(audio.currentTime||0,null) : audio.pause();
};
document.getElementById('playw').onclick = playWord;
document.getElementById('loop').onclick = e=>{ loop=!loop; e.target.textContent = 'loop: '+(loop?'on':'off')+' (L)'; };
document.getElementById('next').onclick = ()=>{ nextWord(); };
document.getElementById('prev').onclick = ()=>{ wi=Math.max(0,wi-1); renderWords(); draw(); };
document.getElementById('skip').onclick = ()=>{ nextUtt(); };
// slower playback makes the moving cursor legible against the waveform, which is how
// you actually judge a boundary; rate must be re-applied after each src change
const rateSel = document.getElementById('rate');
rateSel.onchange = ()=>{ audio.playbackRate = parseFloat(rateSel.value); };
document.getElementById('prevUtt').onclick = ()=>{
  if(ai > 0){ ai--; load(); }
};

function nextWord(){
  const m = marks();
  if(m[wi] && m[wi].s!=null && m[wi].e!=null && wi+1<cur().words.length){
    if(!m[wi+1]) m[wi+1] = {s:m[wi].e, e:null};   // auto-chain for connected speech
  }
  if(wi+1 < cur().words.length){ wi++; renderWords(); draw(); }
  else nextUtt();
}
function nextUtt(){ if(ai+1<DATA.length){ ai++; load(); } else { showJSON(); } }

document.addEventListener('keydown', e=>{
  if(e.target.tagName==='TEXTAREA') return;
  const m = marks();
  if(e.code==='Space'){ e.preventDefault();
    if(e.shiftKey){ playWord(); }
    else { clearTimeout(playTimer);
           audio.paused? playRange(audio.currentTime||0,null) : audio.pause(); } }
  else if(e.key==='Enter'){ e.preventDefault(); nextWord(); }
  else if(e.key==='Backspace'){ e.preventDefault(); wi=Math.max(0,wi-1); renderWords(); draw(); }
  else if(e.key==='ArrowLeft'||e.key==='ArrowRight'){
    e.preventDefault(); if(!m[wi]||m[wi].s==null) return;
    m[wi].s = +Math.max(0,m[wi].s + (e.key==='ArrowRight'?0.01:-0.01)).toFixed(3); persist(); draw();
  }
  else if(e.key===','||e.key==='.'){
    e.preventDefault(); if(!m[wi]||m[wi].e==null) return;
    m[wi].e = +Math.max(0,m[wi].e + (e.key==='.'?0.01:-0.01)).toFixed(3); persist(); draw();
  }
  else if(e.key==='['){ e.preventDefault(); if(ai>0){ ai--; load(); } }
  else if(e.key===']'){ e.preventDefault(); nextUtt(); }
  else if(e.key==='l'||e.key==='L'){ document.getElementById('loop').click(); }
  else if(e.key==='u'||e.key==='U'){ delete marks()[wi]; persist(); renderWords(); draw(); }
});

function collect(){
  const out = [];
  DATA.forEach(d=>{
    const a = ann[d.uid]; if(!a) return;
    const ws = d.words.map((w,i)=> (a[i]&&a[i].s!=null&&a[i].e!=null)
      ? {word:w, start:a[i].s, end:a[i].e} : null);
    if(ws.every(Boolean)) out.push({uid:d.uid, duration:d.duration, words:ws});
  });
  return {batch:"__BATCH__", annotated:out.length, of:DATA.length, utterances:out};
}
function showJSON(){
  const j = JSON.stringify(collect(), null, 1);
  document.getElementById('out').value = j;
  document.getElementById('status').textContent =
    collect().annotated + ' / ' + DATA.length + ' utterances complete';
  document.getElementById('out').select();
}
document.getElementById('save').onclick = showJSON;
document.getElementById('dl').onclick = ()=>{
  const blob = new Blob([JSON.stringify(collect(),null,1)],{type:'application/json'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = 'annotations_batch___BATCH__.json';
  document.body.appendChild(a); a.click(); a.remove();
  document.getElementById('status').textContent =
    'if nothing downloaded, use "finish & show JSON" and copy instead';
};
window.addEventListener('resize', draw);
load();
</script>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned", default="eval_testdataset/aligned_hindi_test.jsonl")
    ap.add_argument("--out", default="annotation")
    ap.add_argument("--n", type=int, default=100, help="unique utterances")
    ap.add_argument("--batches", type=int, default=10,
                    help="total batches; the first half hold copy A, the second half copy B, "
                         "so annotator 1 takes batches 1..N/2 and annotator 2 the rest")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    out = BASE/args.out
    out.mkdir(parents=True, exist_ok=True)
    items = build_items(BASE/args.aligned, args.n, args.batches, args.seed)

    key = {}
    half = args.batches // 2
    for b in range(1, args.batches + 1):
        batch = [it for it in items if it["batch"] == b]
        if not batch: continue
        p_ = out/f"annotate_batch{b:02d}.html"
        p_.write_text(render(batch, b, args.batches), encoding="utf-8")
        print(f"  batch {b:2d} ({'annotator 1' if b <= half else 'annotator 2'}): "
              f"{len(batch):2d} utterances -> {p_.name} ({p_.stat().st_size/1e6:.1f} MB)")
        for it in batch:
            key[it["uid"]] = {"idx": it["idx"], "copy": it["copy"],
                              "batch": b, "text": it["text"]}
    json.dump(key, open(out/"annotation_key.json", "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    uniq = len({v["idx"] for v in key.values()})
    twice = sum(1 for i in {v["idx"] for v in key.values()}
                if sum(1 for v in key.values() if v["idx"] == i) == 2)
    print(f"\n{len(key)} presentations of {uniq} unique utterances; "
          f"{twice} annotated twice (once per annotator)")
    print(f"  annotator 1 -> batches 01..{half:02d}")
    print(f"  annotator 2 -> batches {half+1:02d}..{args.batches:02d}")
    print(f"key -> {out/'annotation_key.json'}  (maps uid -> corpus idx; keep from annotators)")


if __name__ == "__main__":
    main()
