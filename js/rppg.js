/* ============================================================
   AI Eye Scan — rPPG engine
   MediaPipe Face Mesh landmarks -> ROI green-channel sampling ->
   FFT spectral decoding -> BPM / HRV / respiratory rate / perfusion
   SpO2 and blood pressure are deliberately NOT computed: an RGB
   webcam lacks the wavelengths and calibration for them.
   All processing is local (in-memory, never uploaded).
   ============================================================ */
(function () {
  'use strict';

  /* ---------------- small DSP toolkit ---------------- */
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const mean = a => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / (a.length || 1); };
  const std = a => { const m = mean(a); let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2; return Math.sqrt(s / (a.length || 1)); };

  function fft(re, im, inverse) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = 2 * Math.PI / len * (inverse ? 1 : -1);
      const wr = Math.cos(ang), wi = Math.sin(ang), half = len >> 1;
      for (let i = 0; i < n; i += len) {
        let cwr = 1, cwi = 0;
        for (let j = 0; j < half; j++) {
          const ur = re[i + j], ui = im[i + j];
          const xr = re[i + j + half], xi = im[i + j + half];
          const vr = xr * cwr - xi * cwi, vi = xr * cwi + xi * cwr;
          re[i + j] = ur + vr; im[i + j] = ui + vi;
          re[i + j + half] = ur - vr; im[i + j + half] = ui - vi;
          const nwr = cwr * wr - cwi * wi; cwi = cwr * wi + cwi * wr; cwr = nwr;
        }
      }
    }
    if (inverse) for (let i = 0; i < n; i++) { re[i] /= n; im[i] /= n; }
  }

  function hann(n, N) { return 0.5 * (1 - Math.cos(2 * Math.PI * n / (N - 1))); }

  // linear interpolation of an irregular series onto a uniform grid
  function resample(arr, key, t0, dt, count) {
    const out = new Float64Array(count);
    let i = 0;
    for (let k = 0; k < count; k++) {
      const t = t0 + k * dt;
      while (i < arr.length - 2 && arr[i + 1].t < t) i++;
      const a = arr[i], b = arr[i + 1] || a;
      const span = b.t - a.t;
      const f = span > 0 ? clamp((t - a.t) / span, 0, 1) : 0;
      out[k] = a[key] + (b[key] - a[key]) * f;
    }
    return out;
  }

  function detrend(sig, win) {
    const n = sig.length, out = new Float64Array(n);
    const half = win >> 1;
    let sum = 0;
    const cs = new Float64Array(n + 1);
    for (let i = 0; i < n; i++) { sum += sig[i]; cs[i + 1] = sum; }
    for (let i = 0; i < n; i++) {
      const s = Math.max(0, i - half), e = Math.min(n, i + half + 1);
      out[i] = sig[i] - (cs[e] - cs[s]) / (e - s);
    }
    return out;
  }

  // spectrum peak with parabolic interpolation (sub-bin precision)
  function peakOf(mags, sr, N, fMin, fMax) {
    const bin = sr / N;
    const k0 = Math.max(1, Math.ceil(fMin / bin)), k1 = Math.min(mags.length - 2, Math.floor(fMax / bin));
    let best = k0, bv = -Infinity;
    for (let k = k0; k <= k1; k++) if (mags[k] > bv) { bv = mags[k]; best = k; }
    if (bv <= 0) return null;
    const a = Math.log(mags[best - 1] + 1e-12), b = Math.log(mags[best] + 1e-12), c = Math.log(mags[best + 1] + 1e-12);
    const d = (a - 2 * b + c) !== 0 ? 0.5 * (a - c) / (a - 2 * b + c) : 0;
    return { f: (best + clamp(d, -1, 1)) * bin, mag: bv, k: best };
  }

  function bandPower(mags, sr, N, fMin, fMax) {
    const bin = sr / N; let s = 0;
    for (let k = 0; k < mags.length; k++) { const f = k * bin; if (f >= fMin && f <= fMax) s += mags[k] * mags[k]; }
    return s;
  }

  // time-domain peak picking on the band-passed waveform
  function findPeaks(sig, sr, minGapSec) {
    const n = sig.length, m = mean(sig), sd = std(sig);
    if (sd < 1e-9) return [];
    const thr = m + 0.35 * sd, minDist = Math.max(2, Math.floor(minGapSec * sr));
    const peaks = []; let last = -1e9;
    for (let i = 1; i < n - 1; i++) {
      if (sig[i] > thr && sig[i] >= sig[i - 1] && sig[i] > sig[i + 1]) {
        if (i - last < minDist) {
          const p = peaks[peaks.length - 1];
          if (p && sig[i] > p.amp) { p.amp = sig[i]; p.idx = i; p.t = i / sr; }
          continue;
        }
        const a = sig[i - 1], b = sig[i], c = sig[i + 1];
        const den = a - 2 * b + c;
        const off = den !== 0 ? clamp(0.5 * (a - c) / den, -1, 1) : 0;
        peaks.push({ idx: i, t: (i + off) / sr, amp: b });
        last = i;
      }
    }
    return peaks;
  }

  const ema = (prev, next, a) => (prev === null || prev === undefined) ? next : prev + a * (next - prev);
  const fmt = (v, d = 0) => (v === null || v === undefined || !isFinite(v)) ? '--' : v.toFixed(d);

  /* ---------------- the scanner ---------------- */
  class RPPGScanner {
    constructor() {
      this.video = document.getElementById('scanVideo');
      this.overlay = document.getElementById('scanOverlay');
      this.chart = document.getElementById('ppgChart');
      this.octx = this.overlay ? this.overlay.getContext('2d') : null;
      this.cctx = this.chart ? this.chart.getContext('2d') : null;
      this.sample = document.createElement('canvas');
      this.sctx = this.sample.getContext('2d', { willReadFrequently: true });
      this.sample.width = 320;

      this.raw = [];              // [{t,r,g,b,ga,gb}]
      this.trace = [];            // band-passed waveform for the chart
      this.peakMarks = [];
      this.nn = [];               // rolling 60 s of beat intervals for HRV
      this.amp = null;            // measured AC amplitude (a.u.)
      this.pi = null;             // measured perfusion index (%)
      this.unsub = null;
      this.running = false;
      this.stream = null;
      this.mesh = null;
      this.model = 'idle';        // idle | facemesh | fallback
      this.modelErr = 0;
      this.lastLms = null;
      this.lms = null;            // smoothed landmark set
      this.lastRois = null;
      this.faceHits = 0; this.faceTot = 0; this.faceRatio = 0;
      this.lastFaceTime = 0; this.personShown = null;
      this.lastSend = 0; this.lastAnalyze = 0;
      this.sessionStart = 0; this.sessionDone = false;
      this.m = {};                // smoothed metrics
      this.sqi = 0;
      this.busy = false;
      this.raf = 0;
      this.bindUI();
      this.sizeCanvases();
      window.addEventListener('resize', () => this.sizeCanvases());
    }

    /* ---------- DOM helpers ---------- */
    $(id) { return document.getElementById(id); }
    txt(id, v) { const e = this.$(id); if (e) e.textContent = v; }
    cls(id, c, on) { const e = this.$(id); if (e) e.classList.toggle(c, !!on); }

    bindUI() {
      const rs = this.$('btnRestart');
      if (rs) rs.addEventListener('click', () => this.restart());
      const st = this.$('btnStop');
      if (st) st.addEventListener('click', () => this.stop(true));
      // delegate every [data-scan-start] control
      document.querySelectorAll('[data-scan-start]').forEach(b =>
        b.addEventListener('click', () => this.startFlow()));
    }

    sizeCanvases() {
      const dpr = Math.min(2, window.devicePixelRatio || 1);
      if (this.overlay && this.video && this.video.videoWidth) {
        this.overlay.width = this.video.videoWidth;
        this.overlay.height = this.video.videoHeight;
      }
      if (this.chart) {
        const w = this.chart.clientWidth || 400, h = 150;
        this.chart.width = Math.round(w * dpr);
        this.chart.height = Math.round(h * dpr);
        this.cctx && this.cctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      }
    }

    setPerm(state) {
      const label = {
        idle: 'Camera: not requested', requesting: 'Camera: requesting…',
        granted: 'Camera: active', denied: 'Camera: denied'
      }[state];
      ['navPermPill', 'footPermPill'].forEach(id => {
        const e = this.$(id); if (!e) return;
        e.textContent = label;
        e.classList.toggle('granted', state === 'granted');
        e.classList.toggle('denied', state === 'denied');
      });
      const dot = this.$('permDot');
      if (dot) { dot.className = 'dot ' + (state === 'idle' ? '' : state); }
      document.querySelectorAll('.perm-states .state').forEach(s =>
        s.classList.toggle('on', s.dataset.state === state));
      this.txt('badgeCam', state === 'granted' ? 'Camera: live' : label);
      const b = this.$('badgeCam');
      if (b) b.classList.toggle('ok', state === 'granted');
      if (state === 'denied') {
        this.txt('permText', 'Camera access was blocked. Allow the camera in your browser\'s address-bar icon (the camera/lock symbol), then press “Allow camera” again.');
        this.txt('stageEmptyTitle', 'Camera permission denied');
        this.txt('stageEmptyText', 'Open your browser site settings, allow camera access for this page, then retry.');
      }
      if (state === 'requesting') {
        this.txt('stageEmptyTitle', 'Requesting camera…');
        this.txt('stageEmptyText', 'Please accept the browser prompt to begin the scan.');
      }
    }

    /* ---------- camera & model ---------- */
    async startFlow() {
      document.getElementById('rppg')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      if (this.running) return;
      await this.start();
    }

    async start() {
      if (this.running) return;
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        this.txt('stageEmptyTitle', 'No camera API available');
        this.txt('stageEmptyText', 'This browser does not expose getUserMedia. Try Chrome, Edge or Firefox on https/localhost.');
        this.setPerm('denied');
        return;
      }
      this.setPerm('requesting');
      // one camera at a time: release the iris scanner first
      if (window.Iris && window.Iris.running) window.Iris.stop(false);
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
          audio: false
        });
      } catch (err) {
        this.setPerm('denied');
        this.cls('badgeCam', 'bad', true);
        return;
      }
      this.setPerm('granted');
      this.video.srcObject = this.stream;
      try { await this.video.play(); } catch (e) { /* autoplay guard */ }
      await this.initModel();

      this.running = true;
      this.raw = []; this.trace = []; this.peakMarks = [];
      this.nn = []; this.amp = null; this.pi = null;
      this.m = {}; this.faceHits = 0; this.faceTot = 0;
      this.sessionStart = performance.now(); this.sessionDone = false;
      const empty = this.$('stageEmpty');
      if (empty) empty.style.display = 'none';
      this.sizeCanvases();
      this.loop();
    }

    stop(user) {
      this.running = false;
      cancelAnimationFrame(this.raf);
      if (this.stream) { this.stream.getTracks().forEach(t => t.stop()); this.stream = null; }
      this.video.srcObject = null;
      this.setPerm(user ? 'idle' : 'idle');
      const empty = this.$('stageEmpty');
      if (empty) {
        empty.style.display = '';
        this.txt('stageEmptyTitle', 'Camera stopped');
        this.txt('stageEmptyText', 'Session ended. Start again whenever you are ready.');
      }
      this.txt('badgeModel', 'Landmarks: —');
      this.txt('badgeQuality', 'Signal: —');
      this.txt('badgePerson', 'Person: —');
      this.personShown = null; this.lastFaceTime = 0;
      const al = this.$('stageAlert'); if (al) al.hidden = true;
      this.txt('scanHint', 'Sit upright · even lighting · look at the camera · stay still');
    }

    restart() {
      this.raw = []; this.trace = []; this.peakMarks = [];
      this.nn = []; this.amp = null; this.pi = null;
      this.m = {}; this.sessionStart = performance.now(); this.sessionDone = false;
      if (this.running) {
        const sec = Math.floor((performance.now() - this.sessionStart) / 1000);
        this.txt('badgeTimer', '00:00');
      }
      this.txt('scanHint', 'Session restarted — hold still for 60 seconds so HRV can accumulate.');
    }

    async initModel() {
      try {
        if (!window.FaceMeshShared) throw new Error('shared loader missing');
        this.mesh = await window.FaceMeshShared.get();
        if (!this.unsub) {
          this.unsub = window.FaceMeshShared.subscribe(r => {
            this.lastLms = (r.multiFaceLandmarks && r.multiFaceLandmarks[0]) || null;
          });
        }
        this.model = 'facemesh'; this.modelErr = 0;
        this.txt('badgeModel', 'Landmarks: MediaPipe face mesh');
        this.cls('badgeModel', 'ok', true);
        return;
      } catch (e) { this.mesh = null; }
      this.model = 'fallback';
      this.txt('badgeModel', 'Landmarks: colour-ROI fallback');
      this.cls('badgeModel', 'warn', true);
    }

    /* ---------- frame loop ---------- */
    async loop() {
      if (!this.running) return;
      this.raf = requestAnimationFrame(() => this.loop());
      const now = performance.now();
      if (now - this.lastSend < 33 || this.busy) return;
      this.busy = true;
      if (this.mesh && this.video.readyState >= 2) {
        try {
          await window.FaceMeshShared.send(this.mesh, this.video);
          this.modelErr = 0;
        } catch (e) {
          if (++this.modelErr > 40) { this.mesh = null; this.model = 'fallback'; this.txt('badgeModel', 'Landmarks: colour-ROI fallback'); }
        }
      }
      this.busy = false;
      this.lastSend = now;

      this.sampleFrame();
      if (now - this.lastAnalyze > 400) { this.lastAnalyze = now; this.analyze(); }
      this.drawOverlay();
      this.drawChart();
      this.updateSession(now);
      this.updatePerson();
    }

    updateSession(now) {
      const el = (now - this.sessionStart) / 1000;
      const mm = String(Math.floor(el / 60)).padStart(2, '0');
      const ss = String(Math.floor(el % 60)).padStart(2, '0');
      this.txt('badgeTimer', `${mm}:${ss}`);
      const bar = this.$('sessionBar');
      if (bar) bar.style.width = clamp(el / 60 * 100, 0, 100) + '%';
      if (el > 60 && !this.sessionDone) {
        this.sessionDone = true;
        this.txt('scanHint', '✓ 60-second session complete — HRV now has enough beats to report.');
      }
    }

    /* ---------- person-in-frame state ("Person not available") ---------- */
    updatePerson() {
      const hasMesh = this.model === 'facemesh';
      const now = performance.now();
      const elapsed = (now - (this.sessionStart || now)) / 1000;
      let present = true, label = 'Person: in frame', cls = 'ok', alert = false;
      if (hasMesh) {
        if (this.lastFaceTime) {
          present = now - this.lastFaceTime < 1700;
          label = present ? 'Person: in frame' : 'Person: not available';
          cls = present ? 'ok' : 'bad';
          alert = !present;
        } else {
          present = false;
          const searching = elapsed < 4;
          label = searching ? 'Person: searching…' : 'Person: not available';
          cls = searching ? 'warn' : 'bad';
          alert = !searching;
        }
      } else if (this.model === 'fallback') {
        label = 'Person: skin ROI';
        cls = 'warn';
      } else {
        label = 'Person: —';
        cls = '';
      }
      if (this.personShown === label) return;
      this.personShown = label;
      this.txt('badgePerson', label);
      const b = this.$('badgePerson'); if (b) b.className = 'badge ' + cls;
      const a = this.$('stageAlert'); if (a) a.hidden = !alert;
      const h = this.$('scanHint');
      if (h && !this.sessionDone) h.textContent = present
        ? 'Sit upright · even lighting · look at the camera · stay still'
        : 'No person detected — step into the frame and face the camera';
    }

    /* ---------- ROI geometry ---------- */
    roisNorm() {
      const l = this.lastLms;
      if (!l || l.length < 460) {
        this.faceTot++;
        const box = { x: .30, y: .16, w: .40, h: .34 };
        return { rects: [box, box, box, box, box], face: false };
      }
      this.faceHits++;
      this.lastFaceTime = performance.now();
      const W = [], H = [];
      let minx = 1, maxx = 0, miny = 1, maxy = 0;
      for (let i = 0; i < l.length; i += 7) {
        const p = l[i];
        if (p.x < minx) minx = p.x; if (p.x > maxx) maxx = p.x;
        if (p.y < miny) miny = p.y; if (p.y > maxy) maxy = p.y;
      }
      const fw = maxx - minx, fh = maxy - miny;
      const mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });
      const eyeL = mid(l[33], l[133]);
      const eyeR = mid(l[362], l[263]);
      const eyeY = (eyeL.y + eyeR.y) / 2;
      const rect = (cx, cy, w, h) => ({
        x: clamp(cx - w / 2, 0, 1 - w), y: clamp(cy - h / 2, 0, 1 - h), w, h
      });
      this.faceTot++;
      return {
        face: true,
        rects: [
          rect((minx + maxx) / 2, miny + fh * 0.17, fw * 0.56, fh * 0.16),  // forehead
          rect(eyeL.x, eyeY, fw * 0.24, fh * 0.11),                        // left periorbital
          rect(eyeR.x, eyeY, fw * 0.24, fh * 0.11),                        // right periorbital
          rect(minx + fw * 0.30, miny + fh * 0.62, fw * 0.20, fh * 0.15),  // left cheek
          rect(maxx - fw * 0.30, miny + fh * 0.62, fw * 0.20, fh * 0.15)   // right cheek
        ],
        box: { x: minx - fw * .06, y: miny - fh * .06, w: fw * 1.12, h: fh * 1.12 }
      };
    }

    sampleFrame() {
      const v = this.video;
      if (!v || !v.videoWidth) return;
      const W = 320, H = Math.max(1, Math.round(320 * v.videoHeight / v.videoWidth));
      if (this.sample.height !== H) this.sample.height = H;
      this.sctx.drawImage(v, 0, 0, W, H);

      const geo = this.roisNorm();
      this.lastRois = geo;
      const win = this.faceHits + this.faceTot ? this.faceHits / this.faceTot : 0;
      this.faceRatio = ema(this.faceRatio, win, 0.1);
      if (this.faceTot > 120) { this.faceHits = Math.round(this.faceHits * .5); this.faceTot = Math.round(this.faceTot * .5); }

      let img;
      try { img = this.sctx.getImageData(0, 0, W, H); } catch (e) { return; }
      const d = img.data;

      const ch = geo.rects.map(r => {
        const x0 = Math.round(r.x * W), y0 = Math.round(r.y * H);
        const x1 = Math.min(W, Math.round((r.x + r.w) * W)), y1 = Math.min(H, Math.round((r.y + r.h) * H));
        let rs = 0, gs = 0, bs = 0, n = 0;
        for (let y = y0; y < y1; y += 2) {
          let i = (y * W + x0) * 4;
          for (let x = x0; x < x1; x += 2) { rs += d[i]; gs += d[i + 1]; bs += d[i + 2]; i += 8; n++; }
        }
        return n ? { r: rs / n, g: gs / n, b: bs / n } : null;
      }).filter(Boolean);
      if (!ch.length) return;

      // group A = forehead + eyes, group B = cheeks
      const avg = (idx) => {
        let rs = 0, gs = 0, bs = 0, n = 0;
        idx.forEach(i => { const c = ch[i]; if (c) { rs += c.r; gs += c.g; bs += c.b; n++; } });
        return n ? { r: rs / n, g: gs / n, b: bs / n } : null;
      };
      const A = avg([0, 1, 2]) || avg([0]);
      const B = avg([3, 4]) || A;

      this.raw.push({
        t: performance.now() / 1000,
        r: 0.6 * A.r + 0.4 * B.r, g: 0.6 * A.g + 0.4 * B.g, b: 0.6 * A.b + 0.4 * B.b,
        ga: A.g, gb: B.g
      });
      const cutoff = performance.now() / 1000 - 20;
      while (this.raw.length && this.raw[0].t < cutoff) this.raw.shift();
    }

    /* ---------- analysis ---------- */
    analyze() {
      const sr = 30, winSec = 10, N = 512;
      const arr = this.raw;
      if (arr.length < 150) return;
      // never fabricate vitals when no face is being tracked
      if (this.model === 'facemesh' && !(this.lastFaceTime && performance.now() - this.lastFaceTime < 1700)) {
        this.sqi = ema(this.sqi, 0, 0.3);
        this.trace = [];
        this.render(false);
        return;
      }
      const tEnd = arr[arr.length - 1].t;
      const t0 = Math.max(tEnd - winSec, arr[0].t);
      const count = Math.min(winSec * sr, Math.floor((tEnd - t0) * sr));
      if (count < 180) return;

      const grid = t0, dt = 1 / sr;
      const R = resample(arr, 'r', grid, dt, count);
      const G = resample(arr, 'g', grid, dt, count);
      const B = resample(arr, 'b', grid, dt, count);
      const GA = resample(arr, 'ga', grid, dt, count);
      const GB = resample(arr, 'gb', grid, dt, count);

      // ---- spectra ----
      const spec = sig => {
        const re = new Float64Array(N), im = new Float64Array(N);
        for (let i = 0; i < count; i++) re[i] = sig[i];
        // remove DC + slow drift, apply Hann
        const m = mean(re.subarray(0, count));
        for (let i = 0; i < count; i++) re[i] = (re[i] - m) * hann(i, count);
        fft(re, im, false);
        const mags = new Float64Array(N >> 1);
        for (let k = 0; k < (N >> 1); k++) mags[k] = Math.hypot(re[k], im[k]);
        return { re, im, mags };
      };
      const dG = detrend(G, 31);
      const sG = spec(dG);

      // ---- heart rate ----
      const hrPeak = peakOf(sG.mags, sr, N, 0.7, 3.5);
      if (!hrPeak) return;
      const hr = clamp(hrPeak.f * 60, 42, 210);

      // ---- band-passed waveform (ideal brick wall, 0.6–4 Hz) ----
      const bin = sr / N;
      const bre = Float64Array.from(sG.re), bim = Float64Array.from(sG.im);
      for (let k = 0; k <= N / 2; k++) {
        const f = k * bin;
        if (f < 0.6 || f > 4) {
          const k2 = k === 0 || k === N / 2 ? k : N - k;
          bre[k] = bim[k] = 0;
          if (k2 !== k) { bre[k2] = bim[k2] = 0; }
        }
      }
      fft(bre, bim, true);
      const bp = Array.from(bre.subarray(0, count));

      // ---- peaks -> upstroke + rolling 60 s HRV ----
      const peaks = findPeaks(bp, sr, 0.33);
      let upstroke = null;
      if (peaks.length >= 3) {
        const ups = [];
        for (const p of peaks) {
          const a = Math.max(0, p.idx - Math.floor(0.5 * sr));
          let lo = a, lv = bp[a];
          for (let i = a; i < p.idx; i++) if (bp[i] < lv) { lv = bp[i]; lo = i; }
          const ms = (p.idx - lo) / sr * 1000;
          if (ms > 40 && ms < 400) ups.push(ms);
        }
        if (ups.length >= 3) upstroke = mean(ups);
      }
      // Beat intervals are accumulated across the whole session: RMSSD over a
      // 10 s window is statistically meaningless, so nothing is shown until
      // we have at least 20 intervals (~20 beats ≈ 20-30 s, 60 s window).
      if (peaks.length >= 2) {
        for (let i = 1; i < peaks.length; i++) {
          const bt = t0 + peaks[i].t;
          const iv = (peaks[i].t - peaks[i - 1].t) * 1000;
          if (iv < 300 || iv > 2000) continue;
          let dup = false;
          for (let k = this.nn.length - 1; k >= 0 && k >= this.nn.length - 5; k--) {
            if (Math.abs(this.nn[k].t - bt) < 0.2) { dup = true; break; }
          }
          if (!dup) this.nn.push({ t: bt, iv });
        }
        const cutoff = performance.now() / 1000 - 60;
        while (this.nn.length && this.nn[0].t < cutoff) this.nn.shift();
      }
      let hrv = null;
      if (this.nn.length >= 20) {
        const ivs = this.nn.map(x => x.iv);
        let ss = 0, c = 0;
        for (let i = 1; i < ivs.length; i++) { const d = ivs[i] - ivs[i - 1]; ss += d * d; c++; }
        hrv = Math.sqrt(ss / Math.max(1, c));
      }

      // ---- measured pulse amplitude + perfusion index (green channel) ----
      const dcG = mean(G.subarray(0, count));
      const acRms = std(dG.subarray(0, count));
      this.amp = acRms;
      this.pi = clamp(acRms / Math.max(1e-6, dcG) * 100, 0, 12);

      // ---- respiration: 2 s moving-average low-pass, peak in 0.1–0.6 Hz ----
      const lowp = detrend(G, 60);
      const re2 = new Float64Array(N), im2 = new Float64Array(N);
      for (let i = 0; i < count; i++) re2[i] = lowp[i] * hann(i, count);
      fft(re2, im2, false);
      const m2 = new Float64Array(N >> 1);
      for (let k = 0; k < (N >> 1); k++) m2[k] = Math.hypot(re2[k], im2[k]);
      const rPeak = peakOf(m2, sr, N, 0.12, 0.62);
      let rr = rPeak ? clamp(rPeak.f * 60, 6, 32) : null;

      // ---- signal quality ----
      const corr = (() => {
        const a = detrend(GA, 31), b = detrend(GB, 31);
        const ma = mean(a), mb = mean(b);
        let sab = 0, sa = 0, sb = 0;
        for (let i = 0; i < count; i++) {
          const x = a[i] - ma, y = b[i] - mb;
          sab += x * y; sa += x * x; sb += y * y;
        }
        return sab / Math.max(1e-9, Math.sqrt(sa * sb));
      })();
      const bandRatio = bandPower(sG.mags, sr, N, 0.7, 3.5) / Math.max(1e-9, bandPower(sG.mags, sr, N, 0.05, 8));
      const sqi = clamp(0.4 * clamp(corr, 0, 1) + 0.4 * clamp(bandRatio * 5, 0, 1) + 0.2 * this.faceRatio, 0, 1);
      this.sqi = ema(this.sqi, sqi, 0.25);

      // ---- smooth everything ----
      const m = this.m;
      const good = sqi > 0.30 || this.sqi > 0.30;

      // trace for the chart (drawn even on a weak signal)
      this.trace = bp.slice(Math.max(0, count - 270));
      this.peakMarks = peaks.slice(-14).map(p => p.idx - (count - this.trace.length));
      this.lastBeat = performance.now();
      if (!good) { this.render(false); return; }

      m.bpm = ema(m.bpm, hr, 0.18);
      m.hrv = hrv !== null ? ema(m.hrv, hrv, 0.15) : m.hrv;
      m.rr = rr !== null ? ema(m.rr, rr, 0.2) : m.rr;
      m.amp = ema(m.amp, this.amp, 0.12);
      m.pi = ema(m.pi, this.pi, 0.12);
      m.upstroke = upstroke;

      let well = 100;
      well -= Math.max(0, m.bpm - 78) * 0.9 + Math.max(0, 55 - m.bpm) * 0.8;
      if (m.hrv != null) well -= Math.max(0, 45 - m.hrv) * 0.7;
      if (m.rr != null) well -= Math.max(0, Math.abs(m.rr - 15) - 4) * 2;
      m.well = clamp(ema(m.well, clamp(well, 0, 100), 0.2), 0, 100);

      const stress = clamp((100 - m.well) * 0.85 + (m.hrv != null && m.hrv < 40 ? (40 - m.hrv) * 0.6 : 0), 0, 100);
      const fatigue = clamp((100 - m.well) * 0.7 + Math.max(0, m.bpm - 85) * 1.1, 0, 100);
      m.stress = ema(m.stress, stress, 0.25);
      m.fatigue = ema(m.fatigue, fatigue, 0.25);

      this.render(good);
    }

    /* ---------- output ---------- */
    render(good) {
      const m = this.m;
      this.txt('mBpm', fmt(m.bpm, 0));
      this.txt('mHrv', m.hrv != null ? fmt(m.hrv, 0) : '--');
      this.txt('mRr', fmt(m.rr, 1));
      this.txt('mAmp', m.amp != null ? m.amp.toFixed(1) : '--');
      this.txt('mPi', m.pi != null ? m.pi.toFixed(2) : '--');
      this.txt('mWell', fmt(m.well, 0));
      this.txt('chartPeak', `peak: ${fmt(m.bpm, 0)} BPM`);

      const label = m.well >= 80 ? 'Low load' : m.well >= 65 ? 'Balanced load'
        : m.well >= 50 ? 'Elevated load' : 'High load';
      this.txt('mWellSub', `${label} · derived, non-clinical`);

      // honest sub-labels
      this.txt('mHrvSub', m.hrv != null
        ? `RMSSD over ${this.nn.length} intervals · beat-to-beat`
        : `Needs a longer session — ${this.nn.length}/20 intervals`);
      this.txt('mRrSub', 'experimental · breathing modulates the pulse');

      const pct = Math.round(this.sqi * 100);
      const bar = this.$('qBar'); if (bar) bar.style.width = pct + '%';
      this.txt('qText', !good
        ? `Weak signal (${pct}%) — face the light, hold still, avoid backlight`
        : `Signal quality ${pct}% — tracking ${this.model === 'facemesh' ? 'face mesh ROIs' : 'skin ROI (approximate)'}`);
      const bq = this.$('badgeQuality');
      if (bq) {
        bq.textContent = `Signal: ${pct}%`;
        bq.className = 'badge ' + (good ? 'ok' : 'warn');
      }

      // beat pulse on the HR card
      const card = this.$('cardBpm');
      if (card && this.trace.length) {
        card.classList.remove('beat'); void card.offsetWidth; card.classList.add('beat');
      }

      this.renderInferences();
      window.dispatchEvent(new CustomEvent('rppg:update', { detail: { ...m, sqi: this.sqi, good } }));
    }

    /* Only measured quantities here — no mood, sleep or substance guesses. */
    renderInferences() {
      const m = this.m;
      const present = this.model === 'facemesh'
        ? !!(this.lastFaceTime && performance.now() - this.lastFaceTime < 1700)
        : null;
      this.txt('infBpm', m.bpm != null ? `${fmt(m.bpm, 0)} BPM` : '—');
      this.txt('infHrv', m.hrv != null ? `${fmt(m.hrv, 0)} ms` : (this.nn.length ? `${this.nn.length}/20 beats` : 'needs 60 s'));
      this.txt('infRr', m.rr != null ? `${fmt(m.rr, 1)} /min` : '—');
      this.txt('infQuality', `${fmt(this.sqi * 100, 0)}%`);
      this.txt('infFace', present === null ? 'skin ROI' : present ? 'Yes' : 'No');
      const note = document.querySelector('.strip-note');
      if (note) note.textContent = '— measured in this tab only';
    }

    /* ---------- drawing ---------- */
    drawOverlay() {
      const c = this.octx; if (!c || !this.overlay.width) return;
      const W = this.overlay.width, H = this.overlay.height;
      c.clearRect(0, 0, W, H);
      const geo = this.lastRois; if (!geo) return;

      // face box
      if (geo.face && geo.box) {
        c.strokeStyle = 'rgba(52,226,208,.55)';
        c.lineWidth = Math.max(1.5, W / 500);
        c.setLineDash([10, 8]);
        c.strokeRect(geo.box.x * W, geo.box.y * H, geo.box.w * W, geo.box.h * H);
        c.setLineDash([]);
      }
      // ROIs
      geo.rects.forEach((r, i) => {
        const x = r.x * W, y = r.y * H, w = r.w * W, h = r.h * H;
        c.fillStyle = i < 3 ? 'rgba(52,226,208,.16)' : 'rgba(139,124,246,.14)';
        c.strokeStyle = i < 3 ? 'rgba(52,226,208,.9)' : 'rgba(139,124,246,.75)';
        c.lineWidth = Math.max(1.4, W / 640);
        c.beginPath();
        if (c.roundRect) c.roundRect(x, y, w, h, 8); else c.rect(x, y, w, h);
        c.fill(); c.stroke();
      });

      // live pulse marker
      if (this.trace && this.trace.length) {
        const v = this.trace[this.trace.length - 1];
        const r = 8 + Math.abs(v) * 40;
        const cx = W * 0.5, cy = H * 0.06;
        c.beginPath(); c.arc(cx, cy, r, 0, Math.PI * 2);
        c.fillStyle = 'rgba(52,226,208,.35)'; c.fill();
      }
      // corner brackets
      c.strokeStyle = 'rgba(56,189,248,.65)'; c.lineWidth = Math.max(2, W / 400);
      const pad = W * 0.02, len = W * 0.05;
      [[pad, pad, 1, 1], [W - pad, pad, -1, 1], [pad, H - pad, 1, -1], [W - pad, H - pad, -1, -1]].forEach(([x, y, sx, sy]) => {
        c.beginPath();
        c.moveTo(x + sx * len, y); c.lineTo(x, y); c.lineTo(x, y + sy * len);
        c.stroke();
      });
    }

    drawChart() {
      const c = this.cctx; if (!c) return;
      const w = this.chart.clientWidth || 400, h = 150;
      c.clearRect(0, 0, w, h);
      // grid
      c.strokeStyle = 'rgba(140,170,255,.10)'; c.lineWidth = 1;
      for (let i = 1; i < 4; i++) { c.beginPath(); c.moveTo(0, h * i / 4); c.lineTo(w, h * i / 4); c.stroke(); }
      const tr = this.trace;
      if (!tr || tr.length < 10) {
        c.fillStyle = 'rgba(147,160,196,.7)'; c.font = '13px Inter, sans-serif';
        c.fillText('Waiting for camera signal…', 14, h / 2);
        return;
      }
      let mx = 1e-6;
      for (const v of tr) mx = Math.max(mx, Math.abs(v));
      const n = tr.length, step = w / Math.max(1, n - 1);
      const yv = v => h / 2 - (v / mx) * (h * 0.42);

      c.beginPath();
      for (let i = 0; i < n; i++) { const x = i * step, y = yv(tr[i]); i ? c.lineTo(x, y) : c.moveTo(x, y); }
      const grad = c.createLinearGradient(0, 0, w, 0);
      grad.addColorStop(0, 'rgba(56,189,248,.35)');
      grad.addColorStop(1, '#34e2d0');
      c.strokeStyle = grad; c.lineWidth = 2; c.lineJoin = 'round'; c.stroke();

      c.lineTo(w, h); c.lineTo(0, h); c.closePath();
      const fill = c.createLinearGradient(0, 0, 0, h);
      fill.addColorStop(0, 'rgba(52,226,208,.22)');
      fill.addColorStop(1, 'rgba(52,226,208,0)');
      c.fillStyle = fill; c.fill();

      // beat markers
      c.fillStyle = '#fbbf24';
      (this.peakMarks || []).forEach(i => {
        if (i < 0 || i >= n) return;
        c.beginPath(); c.arc(i * step, yv(tr[i]), 3, 0, Math.PI * 2); c.fill();
      });
    }
  }

  window.RPPG = new RPPGScanner();
})();
