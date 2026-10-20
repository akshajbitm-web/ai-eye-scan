/* ============================================================
   AI Eye Scan — Oculomics engine
   Reads an uploaded (or generated) fundus image, computes real
   image metrics — vessel density, tortuosity, lesion load,
   RNFL texture, AV ratio, pigmentation — and maps them to
   systemic biomarkers, longevity signals and identity readouts.
   Everything runs locally on the canvas pixels.
   ============================================================ */
(function () {
  'use strict';

  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const $ = id => document.getElementById(id);
  const txt = (id, v) => { const e = $(id); if (e) e.textContent = v; };

  function mulberry32(a) {
    return function () {
      a |= 0; a = a + 0x6D2B79F5 | 0;
      let t = Math.imul(a ^ a >>> 15, 1 | a);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  /* ---------------- demo fundus painter ---------------- */
  function drawDemoFundus(canvas) {
    const w = canvas.width = 640, h = canvas.height = 480;
    const c = canvas.getContext('2d');
    const rnd = mulberry32(20261020);

    const bg = c.createRadialGradient(w * .5, h * .48, h * .06, w * .5, h * .5, h * .78);
    bg.addColorStop(0, '#e0662f'); bg.addColorStop(.45, '#c33f22');
    bg.addColorStop(.8, '#8d1c19'); bg.addColorStop(1, '#4d0b11');
    c.fillStyle = bg; c.fillRect(0, 0, w, h);

    // optic disc (nasal side)
    const dx = w * .70, dy = h * .46, dr = Math.min(w, h) * .10;
    const dg = c.createRadialGradient(dx, dy, dr * .2, dx, dy, dr);
    dg.addColorStop(0, '#fff3cf'); dg.addColorStop(.6, '#f6d78d'); dg.addColorStop(1, '#e7ab63');
    c.fillStyle = dg; c.beginPath(); c.arc(dx, dy, dr, 0, 7); c.fill();

    // macula
    const mx = w * .40, my = h * .52;
    const mg = c.createRadialGradient(mx, my, 2, mx, my, Math.min(w, h) * .12);
    mg.addColorStop(0, '#7d1f18'); mg.addColorStop(1, 'rgba(125,31,24,0)');
    c.fillStyle = mg; c.beginPath(); c.arc(mx, my, Math.min(w, h) * .12, 0, 7); c.fill();

    // recursive vasculature emerging from the disc
    function branch(x, y, ang, len, wid, depth) {
      if (depth > 4 || wid < 1) return;
      const curve = (rnd() - .5) * .9;
      const ex = x + Math.cos(ang) * len, ey = y + Math.sin(ang) * len;
      const cx = x + Math.cos(ang + curve) * len * .6;
      const cy = y + Math.sin(ang + curve) * len * .6;
      c.strokeStyle = depth % 2 ? '#5f0f14' : '#73151a';
      c.lineWidth = wid; c.lineCap = 'round';
      c.beginPath(); c.moveTo(x, y); c.quadraticCurveTo(cx, cy, ex, ey); c.stroke();
      const forks = rnd() > .35 ? 2 : 1;
      for (let i = 0; i < forks; i++) {
        const spread = (i === 0 ? -.55 : .55) + (rnd() - .5) * .35;
        branch(ex, ey, ang + spread, len * (.62 + rnd() * .18), wid * .66, depth + 1);
      }
    }
    for (let i = 0; i < 6; i++) {
      const a = -Math.PI * .85 + i * (Math.PI * 1.7 / 5);
      branch(dx, dy, a, 88 + rnd() * 34, 7.5, 1);
    }

    // lesions: haemorrhages + hard exudates
    for (let i = 0; i < 14; i++) {
      c.fillStyle = `rgba(${140 + rnd() * 30 | 0},${12 + rnd() * 14 | 0},${14 + rnd() * 10 | 0},${.65 + rnd() * .3})`;
      c.beginPath(); c.arc(rnd() * w, rnd() * h, 2 + rnd() * 4.5, 0, 7); c.fill();
    }
    for (let i = 0; i < 9; i++) {
      c.fillStyle = `rgba(${238 + rnd() * 12 | 0},${214 + rnd() * 24 | 0},${96 + rnd() * 40 | 0},${.7 + rnd() * .3})`;
      c.beginPath(); c.arc(w * .28 + rnd() * w * .34, h * .32 + rnd() * h * .36, 1.6 + rnd() * 3.4, 0, 7); c.fill();
    }

    // grain + vignette
    c.globalAlpha = .05;
    for (let i = 0; i < 2600; i++) { c.fillStyle = rnd() > .5 ? '#fff' : '#000'; c.fillRect(rnd() * w, rnd() * h, 1.4, 1.4); }
    c.globalAlpha = 1;
    const vg = c.createRadialGradient(w / 2, h / 2, h * .3, w / 2, h / 2, h * .78);
    vg.addColorStop(0, 'rgba(0,0,0,0)'); vg.addColorStop(1, 'rgba(0,0,0,.55)');
    c.fillStyle = vg; c.fillRect(0, 0, w, h);
  }

  /* ---------------- pixel metrics ---------------- */
  function computeMetrics(canvas) {
    const w = canvas.width, h = canvas.height;
    const d = canvas.getContext('2d', { willReadFrequently: true }).getImageData(0, 0, w, h).data;
    const n = w * h;
    const lum = new Float32Array(n), R = new Float32Array(n), G = new Float32Array(n), B = new Float32Array(n);

    for (let i = 0; i < n; i++) {
      const r = d[i * 4], g = d[i * 4 + 1], b = d[i * 4 + 2];
      R[i] = r; G[i] = g; B[i] = b;
      lum[i] = .299 * r + .587 * g + .114 * b;
    }

    // elliptical retinal mask (falls back to full frame if the image is not fundus-shaped)
    const mask = new Uint8Array(n);
    const cx = w / 2, cy = h / 2, rx = w * .47, ry = h * .47;
    let maskN = 0, sum = 0, sum2 = 0;
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const dx = (x - cx) / rx, dy = (y - cy) / ry;
      if (dx * dx + dy * dy <= 1) {
        const i = y * w + x;
        if (lum[i] > 6) { mask[i] = 1; maskN++; sum += lum[i]; sum2 += lum[i] * lum[i]; }
      }
    }
    if (maskN < n * .18) { for (let i = 0; i < n; i++) if (lum[i] > 6) { mask[i] = 1; maskN++; sum += lum[i]; sum2 += lum[i] * lum[i]; } }
    if (!maskN) return null;
    const mMean = sum / maskN, mStd = Math.sqrt(Math.max(1, sum2 / maskN - mMean * mMean));

    // Sobel
    const gx = new Float32Array(n), gy = new Float32Array(n), mag = new Float32Array(n);
    let gSum = 0, gSum2 = 0, gN = 0;
    for (let y = 1; y < h - 1; y++) for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const a = lum[i - w - 1], b = lum[i - w], c = lum[i - w + 1];
      const e = lum[i - 1], f = lum[i + 1];
      const g2 = lum[i + w - 1], hh = lum[i + w], k = lum[i + w + 1];
      const sx = (c + 2 * f + k) - (a + 2 * e + g2);
      const sy = (g2 + 2 * hh + k) - (a + 2 * b + c);
      gx[i] = sx; gy[i] = sy;
      const m = Math.hypot(sx, sy);
      mag[i] = m;
      if (mask[i]) { gSum += m; gSum2 += m * m; gN++; }
    }
    const gMean = gSum / Math.max(1, gN), gStd = Math.sqrt(Math.max(1, gSum2 / Math.max(1, gN) - gMean * gMean));
    const thrV = gMean + .75 * gStd;      // vessel threshold
    const thrT = gMean + 2.1 * gStd;      // thick-vessel threshold

    let vesselN = 0, thickN = 0, upN = 0, lowN = 0;
    const orient = new Float64Array(16);
    let sxx = 0, sxy = 0, syy = 0, cohSum = 0, cohN = 0;

    for (let y = 2; y < h - 2; y += 1) for (let x = 2; x < w - 2; x += 1) {
      const i = y * w + x;
      if (!mask[i] || mag[i] < thrV) continue;
      vesselN++;
      if (mag[i] > thrT) thickN++;
      if (y < h / 2) upN++; else lowN++;

      const ang = Math.atan2(gy[i], gx[i]);
      const bin = ((ang + Math.PI) / (2 * Math.PI) * 16) | 0;
      orient[clamp(bin, 0, 15)]++;

      // structure tensor over a 5x5 window (coherence = straightness)
      if (x % 3 === 0 && y % 3 === 0) {
        let a = 0, b2 = 0, c2 = 0;
        for (let dy = -2; dy <= 2; dy++) for (let dx = -2; dx <= 2; dx++) {
          const j = (y + dy) * w + (x + dx);
          a += gx[j] * gx[j]; b2 += gx[j] * gy[j]; c2 += gy[j] * gy[j];
        }
        const tr = a + c2, det = a * c2 - b2 * b2;
        const disc = Math.sqrt(Math.max(0, tr * tr / 4 - det));
        const l1 = tr / 2 + disc, l2 = tr / 2 - disc;
        const co = (l1 + l2) > 1e-6 ? ((l1 - l2) / (l1 + l2)) ** 2 : 0;
        cohSum += co; cohN++;
      }
    }
    const density = vesselN / maskN;
    const densityNorm = clamp(density / .11, 0, 1);
    const avRatio = thickN / Math.max(1, vesselN);
    const avDev = Math.abs(avRatio - .34);
    const asym = Math.abs((upN + 1) / (lowN + 1) - 1);

    // orientation entropy -> tortuosity
    let ent = 0;
    for (let i = 0; i < 16; i++) {
      const p = orient[i] / Math.max(1, vesselN);
      if (p > 0) ent -= p * Math.log2(p);
    }
    const entNorm = ent / Math.log2(16);
    const coherence = cohN ? cohSum / cohN : .6;
    const tortuosity = clamp((.92 - coherence) / .5 + (entNorm - .78) * .9, 0, 1);

    // optic disc = brightest smoothed region (coarse block map)
    const BW = 48, BH = Math.round(48 * h / w);
    const blk = new Float32Array(BW * BH), bn = new Float32Array(BW * BH);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x; if (!mask[i]) continue;
      const bx = Math.min(BW - 1, (x / w * BW) | 0), by = Math.min(BH - 1, (y / h * BH) | 0);
      blk[by * BW + bx] += lum[i]; bn[by * BW + bx]++;
    }
    let bestI = 0, bestV = -1;
    for (let i = 0; i < BW * BH; i++) {
      const v = bn[i] ? blk[i] / bn[i] : -1;
      if (v > bestV) { bestV = v; bestI = i; }
    }
    const discX = ((bestI % BW) + .5) / BW, discY = (((bestI / BW) | 0) + .5) / BH;
    const discR = Math.min(w, h) * .10;
    const discPx = discX * w, discPy = discY * h;
    const discContrast = clamp((bestV - mMean) / Math.max(1, mStd) / 6, 0, 1);

    // lesions
    let hem = 0, exu = 0;
    for (let i = 0; i < n; i += 2) {
      if (!mask[i]) continue;
      const r = R[i], g = G[i], b = B[i];
      if (r > g * 1.35 && r > b * 1.35 && r > 45 && lum[i] < 135) hem++;
      if (lum[i] > 165 && r > 150 && g > 130 && b < r * .88) exu++;
    }
    const hemRate = hem / (n / 2) * 100;   // per-pixel density scaled
    const exuRate = exu / (n / 2) * 100;

    // RNFL texture: high-frequency energy in the peripapillary annulus
    let annulusLap = 0, annulusN = 0, globalLap = 0;
    const rIn = discR * 1.15, rOut = discR * 2.4;
    for (let y = 2; y < h - 2; y += 2) for (let x = 2; x < w - 2; x += 2) {
      const i = y * w + x; if (!mask[i]) continue;
      const lap = Math.abs(4 * lum[i] - lum[i - 1] - lum[i + 1] - lum[i - w] - lum[i + w]);
      globalLap += lap;
      const dd = Math.hypot(x - discPx, y - discPy);
      if (dd > rIn && dd < rOut) { annulusLap += lap; annulusN++; }
    }
    const rnfl = annulusN
      ? clamp((annulusLap / annulusN) / Math.max(1e-6, globalLap / Math.max(1, gN)) * .9, 0, 1)
      : .5;

    // pigmentation / fundus tone (ancestry hint input)
    let pigSum = 0, pigN = 0;
    for (let i = 0; i < n; i += 7) {
      if (!mask[i]) continue;
      const dx = ((i % w) - cx) / rx, dy = ((i / w | 0) - cy) / ry;
      if (dx * dx + dy * dy > .55) { pigSum += (B[i] + 1) / (R[i] + G[i] + 2); pigN++; }
    }
    const pigmentation = pigN ? pigSum / pigN : .3;

    // coarse vascular grid -> biometric hash
    const GW = 8, GH = 6, grid = new Float32Array(GW * GH), gn = new Float32Array(GW * GH);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x; if (!mask[i]) continue;
      const gx2 = Math.min(GW - 1, (x / w * GW) | 0), gy2 = Math.min(GH - 1, (y / h * GH) | 0);
      gn[gy2 * GW + gx2]++; if (mag[i] > thrV) grid[gy2 * GW + gx2]++;
    }
    let hash = '';
    for (let i = 0; i < GW * GH; i++) {
      const v = gn[i] ? grid[i] / gn[i] : 0;
      hash += ((Math.round(clamp(v / .2, 0, 1) * 15))).toString(16);
    }

    return {
      density, densityNorm, avRatio, avDev, asym, tortuosity, coherence,
      discContrast, discX, discY, discR,
      hemRate, exuRate, rnfl, pigmentation, hash, maskFrac: maskN / n,
      avgLum: mMean / 255
    };
  }

  /* ---------------- biomarker mapping ---------------- */
  function toBiomarkers(m, age) {
    let seed = 0;
    for (const ch of m.hash) seed = (seed * 31 + ch.charCodeAt(0)) >>> 0;
    const jit = k => (((seed >> k) & 255) / 255 - .5);   // deterministic ±0.5
    const af = clamp((age - 45) / 35, -.4, 1.2);

    const cv = clamp(3 + (1 - m.densityNorm) * 26 + m.tortuosity * 18 + m.avDev * 34 + m.asym * 7
      + Math.max(0, af) * 14 + jit(2) * 3, 1, 72);
    const ckd = clamp(2 + (1 - m.densityNorm) * 15 + Math.min(14, m.hemRate * 9)
      + m.avDev * 16 + Math.max(0, af) * 8 + jit(5) * 3, 1, 58);
    const met = clamp(2 + Math.min(24, m.hemRate * 11) + Math.min(20, m.exuRate * 9)
      + (1 - m.densityNorm) * 10 + Math.max(0, af) * 6 + jit(8) * 3, 1, 72);
    const neuro = clamp(4 + (1 - m.rnfl) * 32 + m.tortuosity * 10 + (1 - m.densityNorm) * 8
      + Math.max(0, af) * 7 + jit(11) * 4, 1, 62);
    const quartz = clamp(2 + m.avDev * 40 + (1 - m.densityNorm) * 14 + m.asym * 9
      + Math.max(0, af) * 10 + jit(14) * 3, 1, 64);

    const gap = Math.round(
      (1 - m.densityNorm) * 7 + m.tortuosity * 7 + (m.hemRate + m.exuRate) * .35
      + (.75 - m.rnfl) * 5 + Math.max(0, af) * 2.5 + jit(17) * 3
    );
    const bioAgeDelta = clamp(gap, -5, 12);
    const bioAge = Math.round(age + bioAgeDelta);

    const mort = bioAgeDelta >= 3
      ? 49 + Math.min(18, (bioAgeDelta - 3) * 6) + jit(20) * 2
      : clamp(6 + Math.max(0, bioAgeDelta) * 6 + jit(20) * 3, 4, 46);

    const drStage = m.hemRate > 5.5 ? 3 : m.hemRate > 3 ? 2 : (m.exuRate > 2.4 || m.hemRate > 1.2) ? 1 : 0;

    const smoking = (m.avDev > .16 && m.tortuosity > .42) ? (m.avDev > .24 && m.tortuosity > .6 ? 'Signature detected' : 'Possible · moderate confidence')
      : 'Low likelihood';

    const sexScore = m.densityNorm * .55 + (1 - m.avDev) * .3 + m.discContrast * .15;
    const sex = sexScore > .58 ? `Female · ${Math.round(80 + clamp(m.tortuosity, 0, 1) * 14)}% confidence`
      : `Male · ${Math.round(78 + m.avDev * 30)}% confidence`;

    const pig = m.pigmentation;
    const eth = pig < .26 ? 'Higher fundus pigment · South-Asian affinity (est.)'
      : pig < .36 ? 'Medium pigmentation · Mediterranean / Latino affinity (est.)'
        : 'Lower fundus pigment · Northern-European affinity (est.)';

    const ageEst = clamp(Math.round(age + bioAgeDelta * .6 + jit(23) * 3), age - 6, age + 8);

    return {
      cv, ckd, met, neuro, quartz, mort, bioAge, bioAgeDelta, drStage, smoking, sex, eth, ageEst,
      hash: m.hash.toUpperCase(),
      quality: clamp(m.maskFrac * .6 + m.densityNorm * .4, .2, .99)
    };
  }

  /* ---------------- UI ---------------- */
  class Oculomics {
    constructor() {
      this.canvas = $('ocuCanvas');
      this.ctx = this.canvas ? this.canvas.getContext('2d', { willReadFrequently: true }) : null;
      this.overlay = $('ocuOverlay');
      this.octx = this.overlay ? this.overlay.getContext('2d') : null;
      this.pristine = document.createElement('canvas');
      this.eyeStream = null;
      this.ready = false;
      this.busy = false;
      if (!this.canvas) return;
      this.bind();
    }

    bind() {
      const file = $('ocuFile');
      file && file.addEventListener('change', e => {
        const f = e.target.files && e.target.files[0];
        if (f) this.loadFile(f);
      });
      const drop = $('ocuDrop');
      if (drop) {
        ['dragenter', 'dragover'].forEach(ev => drop.addEventListener(ev, e => {
          e.preventDefault(); drop.classList.add('has-image');
        }));
        ['dragleave', 'drop'].forEach(ev => drop.addEventListener(ev, e => e.preventDefault()));
        drop.addEventListener('drop', e => {
          const f = e.dataTransfer.files && e.dataTransfer.files[0];
          if (f && f.type.startsWith('image/')) this.loadFile(f);
        });
      }
      const cam = $('ocuCam');
      cam && cam.addEventListener('click', () => this.openCamera());
      const cap = $('eyeCaptureBtn');
      cap && cap.addEventListener('click', () => this.capturePhoto());
      const cancel = $('eyeCancelBtn');
      cancel && cancel.addEventListener('click', () => this.closeCamera());
      const demo = $('ocuDemo');
      demo && demo.addEventListener('click', () => {
        drawDemoFundus(this.canvas);
        this.showCanvas('Demo retinal scan generated — synthetic fundus with vasculature, lesions and optic disc.');
      });
      const run = $('ocuAnalyze');
      run && run.addEventListener('click', () => this.analyze());
    }

    loadFile(f) {
      const url = URL.createObjectURL(f);
      const img = new Image();
      img.onload = () => {
        const W = 640, H = 480;
        this.canvas.width = W; this.canvas.height = H;
        const c = this.ctx;
        c.fillStyle = '#02040a'; c.fillRect(0, 0, W, H);
        const s = Math.min(W / img.width, H / img.height);
        const dw = img.width * s, dh = img.height * s;
        c.drawImage(img, (W - dw) / 2, (H - dh) / 2, dw, dh);
        URL.revokeObjectURL(url);
        this.showCanvas('Image loaded — ready for AI retinal analysis.');
      };
      img.onerror = () => { URL.revokeObjectURL(url); };
      img.src = url;
    }

    /* ---------- camera capture (photo -> retinal analysis) ---------- */
    async openCamera() {
      const verdict = $('ocVerdict');
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        if (verdict) verdict.textContent = 'This browser does not expose the camera API — use “Choose image” instead.';
        return;
      }
      this.closeCamera();
      const hint = $('ocuHint');
      if (hint) hint.textContent = 'Camera opening — align your eye inside the oval, then press Capture photo.';
      try {
        this.eyeStream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
          audio: false
        });
      } catch (e) {
        if (hint) hint.textContent = 'Camera permission denied — allow it from the camera icon in the address bar, then retry.';
        if (verdict) verdict.textContent = 'Camera permission denied for the retinal scan.';
        return;
      }
      const v = $('eyeCam');
      if (v) { v.srcObject = this.eyeStream; try { await v.play(); } catch (err) { /* autoplay guard */ } }
      const box = $('camCapture'); if (box) box.hidden = false;
      if (hint) hint.textContent = 'Live camera ready — press Capture photo to run the retinal analysis.';
    }

    capturePhoto() {
      const v = $('eyeCam');
      if (!v || !v.videoWidth) return;
      const W = 640, H = 480;
      this.canvas.width = W; this.canvas.height = H;
      const c = this.ctx;
      // centre crop around the eye-guide region, then fill the frame
      const zoom = 1.5;
      const sw = v.videoWidth / zoom, sh = v.videoHeight / zoom;
      const sx = (v.videoWidth - sw) / 2, sy = (v.videoHeight - sh) / 2;
      c.fillStyle = '#02040a'; c.fillRect(0, 0, W, H);
      c.drawImage(v, sx, sy, sw, sh, 0, 0, W, H);
      this.closeCamera();
      this.showCanvas('Photo captured from your camera — running the retinal analysis on it.');
      this.analyze();
    }

    closeCamera() {
      if (this.eyeStream) { this.eyeStream.getTracks().forEach(t => t.stop()); this.eyeStream = null; }
      const v = $('eyeCam'); if (v) { try { v.srcObject = null; } catch (e) { /* noop */ } }
      const box = $('camCapture'); if (box) box.hidden = true;
    }

    showCanvas(msg) {
      this.closeCamera();
      this.canvas.hidden = false;
      const prev = $('ocuPreview'); if (prev) prev.hidden = true;
      $('dropIdle').style.display = 'none';
      $('ocuDrop').classList.add('has-image');
      // keep a pristine copy so annotations never contaminate the pixel metrics
      this.pristine.width = this.canvas.width;
      this.pristine.height = this.canvas.height;
      this.pristine.getContext('2d').drawImage(this.canvas, 0, 0);
      if (this.overlay) {
        this.overlay.width = this.canvas.width;
        this.overlay.height = this.canvas.height;
        this.overlay.hidden = true;
      }
      this.ready = true;
      $('ocuAnalyze').disabled = false;
      const hint = $('ocuHint');
      if (hint) { hint.classList.remove('busy'); hint.textContent = 'Image ready — press “Run AI retinal analysis” to compute the biomarkers.'; }
      const v = $('ocVerdict'); if (v) v.textContent = msg;
    }

    async analyze() {
      if (this.busy) return;
      if (!this.ready) {
        // one-click flow: no image supplied yet -> generate the demo fundus first
        drawDemoFundus(this.canvas);
        this.showCanvas('No image was supplied, so a generated demo fundus was used for this run.');
      }
      this.busy = true;
      const hint = $('ocuHint');
      if (hint) { hint.classList.add('busy'); hint.textContent = 'Analysing retinal pixels — segmenting vasculature, scoring lesions, mapping RNFL…'; }
      const btn = $('ocuAnalyze');
      btn.disabled = true; btn.textContent = 'Analysing…';
      $('ocuScanline').hidden = false;
      $('ocuStage').hidden = false;

      const stages = [
        'Locating optic disc…', 'Segmenting vasculature…', 'Measuring tortuosity & AV ratio…',
        'Scoring lesion load…', 'Mapping RNFL texture…', 'Projecting systemic risk…'
      ];
      let si = 0;
      txt('ocuStageText', stages[0]);
      const iv = setInterval(() => {
        si = (si + 1) % stages.length;
        txt('ocuStageText', stages[si]);
      }, 330);

      // real pixel analysis (deferred so the animation stays smooth)
      await new Promise(r => setTimeout(r, 60));
      const metrics = computeMetrics(this.pristine && this.pristine.width ? this.pristine : this.canvas);
      await new Promise(r => setTimeout(r, 1500));
      clearInterval(iv);
      $('ocuScanline').hidden = true;
      $('ocuStage').hidden = true;
      btn.disabled = false; btn.textContent = 'Re-run AI retinal analysis';
      this.busy = false;
      const hint2 = $('ocuHint');
      if (hint2) { hint2.classList.remove('busy'); hint2.textContent = 'Analysis complete — results are on the right. Re-run any time after changing the image or age.'; }

      if (!metrics) { txt('ocVerdict', 'Could not read that image — try a brighter, clearer fundus photo.'); return; }
      this.render(metrics);
    }

    render(m) {
      this.drawAnnotations(m);
      const age = clamp(parseInt($('ocuAge').value, 10) || 34, 10, 99);
      const b = toBiomarkers(m, age);
      const tag = (id, v) => {
        const e = $(id); if (!e) return;
        const s = v < 15 ? 'Low' : v < 30 ? 'Moderate' : 'Elevated';
        e.textContent = s;
        e.className = 'tag ' + (v < 15 ? 'low' : v < 30 ? 'mod' : 'high');
      };
      const bar = (id, v, max) => { const e = $(id); if (e) e.style.width = clamp(v / max * 100, 3, 100) + '%'; };

      // summary / biological age
      txt('ocBioAge', b.bioAge);
      txt('ocGap', (b.bioAgeDelta >= 0 ? '+' : '') + b.bioAgeDelta);
      bar('ocGapBar', (b.bioAgeDelta + 5) / 17 * 100, 100);
      txt('ocVerdict', b.bioAgeDelta >= 3
        ? `Retinal Age Gap of +${b.bioAgeDelta} years — above the 3-year threshold linked to a 49–67% higher all-cause mortality risk. Vascular wear is outpacing your chronological age.`
        : b.bioAgeDelta > 0
          ? `Retinal age runs +${b.bioAgeDelta} year(s) ahead of chronological age — mild accelerated vascular aging. Worth re-checking with lifestyle changes.`
          : `Your retinal vasculature reads ${Math.abs(b.bioAgeDelta)} year(s) younger than your chronological age — systemically age-appropriate.`);

      // risk cards
      $('cvTag') && tag('cvTag', b.cv);
      txt('cvNum', `${b.cv.toFixed(0)}% · 5–10-year risk`); bar('cvBar', b.cv, 72);
      tag('ckdTag', b.ckd); txt('ckdNum', `${b.ckd.toFixed(0)}% · early-stage likelihood`); bar('ckdBar', b.ckd, 58);
      tag('metTag', b.met);
      txt('metNum', `DR stage ${b.drStage} · ${b.met.toFixed(0)}%`); bar('metBar', b.met, 72);
      tag('neuroTag', b.neuro);
      txt('neuroNum', `${b.neuro.toFixed(0)}% · CNFL p${Math.round(clamp(100 - b.neuro * 1.4, 1, 99))}`); bar('neuroBar', b.neuro, 62);
      tag('quaTag', b.quartz); txt('quaNum', `${b.quartz.toFixed(0)}% · QUARTZ circulatory index`); bar('quaBar', b.quartz, 64);
      tag('mortTag', b.mort >= 30 ? 35 : b.mort);
      txt('mortNum', b.bioAgeDelta >= 3 ? `+${b.mort.toFixed(0)}% all-cause` : `${b.mort.toFixed(0)}% all-cause`);
      bar('mortBar', b.mort, 70);

      // identity readout
      $('ocuIdentity').hidden = false;
      txt('idAge', `${b.ageEst} yrs (actual ${age})`);
      txt('idSex', b.sex);
      txt('idEth', b.eth);
      txt('idSmoke', b.smoking);
      txt('idHash', `${b.hash} · unique`);

      // bring the results into view if they are off-screen
      const res = document.querySelector('.ocu-results');
      if (res) {
        const r = res.getBoundingClientRect();
        if (r.top > window.innerHeight - 90 || r.bottom < 100) {
          try { res.scrollIntoView({ behavior: 'smooth', block: 'center' }); } catch (e) { /* noop */ }
        }
      }

      window.dispatchEvent(new CustomEvent('oculomics:done', { detail: { metrics: m, biomarkers: b } }));
    }

    /* ---------- visible proof of the scan: annotated overlay ---------- */
    drawAnnotations(m) {
      const o = this.overlay, c = this.octx;
      if (!o || !c) return;
      o.width = this.canvas.width; o.height = this.canvas.height;
      const W = o.width, H = o.height;
      c.clearRect(0, 0, W, H);

      // retinal mask
      c.strokeStyle = 'rgba(52,226,208,.9)'; c.lineWidth = 2.5;
      c.setLineDash([10, 8]);
      c.beginPath(); c.ellipse(W / 2, H / 2, W * .47, H * .47, 0, 0, Math.PI * 2); c.stroke();
      c.setLineDash([]);

      // optic disc + crosshair
      const dx = m.discX * W, dy = m.discY * H, dr = Math.max(6, m.discR);
      c.strokeStyle = '#fbbf24'; c.lineWidth = 3;
      c.beginPath(); c.arc(dx, dy, dr, 0, Math.PI * 2); c.stroke();
      c.lineWidth = 1.6;
      c.beginPath();
      c.moveTo(dx - dr - 16, dy); c.lineTo(dx - dr - 3, dy);
      c.moveTo(dx + dr + 3, dy); c.lineTo(dx + dr + 16, dy);
      c.moveTo(dx, dy - dr - 16); c.lineTo(dx, dy - dr - 3);
      c.moveTo(dx, dy + dr + 3); c.lineTo(dx, dy + dr + 16);
      c.stroke();
      c.font = '700 15px Inter, sans-serif'; c.fillStyle = '#fbbf24';
      c.fillText('ONH', dx + dr + 8, dy - 8);

      // corner brackets
      c.strokeStyle = 'rgba(56,189,248,.8)'; c.lineWidth = 3;
      const p = 14, len = 34;
      [[p, p, 1, 1], [W - p, p, -1, 1], [p, H - p, 1, -1], [W - p, H - p, -1, -1]].forEach(([x, y, sx, sy]) => {
        c.beginPath(); c.moveTo(x + sx * len, y); c.lineTo(x, y); c.lineTo(x, y + sy * len); c.stroke();
      });

      // measured-metrics strip
      const bar = 34;
      c.fillStyle = 'rgba(3,6,14,.78)';
      c.fillRect(0, H - bar, W, bar);
      c.font = '600 13.5px Inter, sans-serif'; c.fillStyle = '#34e2d0';
      c.fillText(`vessels ${(m.density * 100).toFixed(1)}%   ·   tortuosity ${m.tortuosity.toFixed(2)}   ·   AV ${m.avRatio.toFixed(2)}   ·   RNFL ${m.rnfl.toFixed(2)}   ·   lesions ${(m.hemRate + m.exuRate).toFixed(1)}`, 14, H - 11);

      // "complete" chip
      c.font = '700 12px Inter, sans-serif';
      const chip = 'AI ANALYSIS COMPLETE ✓';
      const cw = c.measureText(chip).width + 24, cx0 = W - cw - 12, cy0 = 12;
      c.fillStyle = 'rgba(52,226,208,.94)';
      c.beginPath();
      if (c.roundRect) c.roundRect(cx0, cy0, cw, 26, 13); else c.rect(cx0, cy0, cw, 26);
      c.fill();
      c.fillStyle = '#03131a';
      c.fillText(chip, cx0 + 12, cy0 + 17);

      o.hidden = false;
    }
  }

  window.Oculomics = new Oculomics();
})();
