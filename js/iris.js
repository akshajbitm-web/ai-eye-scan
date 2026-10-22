/* ============================================================
   AI Eye Scan — Iris module
   Real, on-device measurements from a webcam iris image:

     · iris localisation from MediaPipe iris landmarks (468-477)
     · rubber-sheet polar unwrap (pupil boundary -> limbus)
     · Daugman-style Gabor iris code + rotation-tolerant matching
     · iris colour, pupil/iris ratio, crypt & pigment-spot counts
     · conservative ocular-sign screening flags (never diagnoses)

   Everything is computed in this tab. Nothing is uploaded.
   ============================================================ */
(function () {
  'use strict';

  /* ---------------- helpers ---------------- */
  const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
  const $ = id => document.getElementById(id);
  const txt = (id, v) => { const e = $(id); if (e) e.textContent = v; };
  const mean = a => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return s / (a.length || 1); };
  const std = a => { const m = mean(a); let s = 0; for (let i = 0; i < a.length; i++) s += (a[i] - m) ** 2; return Math.sqrt(s / (a.length || 1)); };

  function rgb2hsv(r, g, b) {
    r /= 255; g /= 255; b /= 255;
    const mx = Math.max(r, g, b), mn = Math.min(r, g, b), d = mx - mn;
    let h = 0;
    if (d > 1e-6) {
      if (mx === r) h = ((g - b) / d) % 6;
      else if (mx === g) h = (b - r) / d + 2;
      else h = (r - g) / d + 4;
      h *= 60; if (h < 0) h += 360;
    }
    return { h, s: mx === 0 ? 0 : d / mx, v: mx };
  }

  function circMean(hues) {
    let x = 0, y = 0;
    for (const h of hues) { const a = h * Math.PI / 180; x += Math.cos(a); y += Math.sin(a); }
    let m = Math.atan2(y, x) * 180 / Math.PI;
    if (m < 0) m += 360;
    return m;
  }
  function circStd(hues, m) {
    if (!hues.length) return 180;
    let s = 0;
    for (const h of hues) { let d = Math.abs(h - m); if (d > 180) d = 360 - d; s += d * d; }
    return Math.sqrt(s / hues.length);
  }

  /* ---------------- geometry / constants ---------------- */
  const CROP = 320;
  const PAD = 1.42;                       // limbus sits at CROP/2/PAD
  const IRIS_R = CROP / 2 / PAD;          // ~112.7 px in the crop
  const UW = 512, UR = 64;                // unwrap grid (angle x radius)
  const CW = 128, CH = 20;                // code grid (angle x radius)
  const SCALES = [1.1, 1.7, 2.6];
  const ORIENTS = [0, 30, 60, 90, 120, 150];
  const PER = SCALES.length * ORIENTS.length * 2;
  const GENUINE_MAX = 0.35;               // Hamming distance
  const INCONCLUSIVE_MAX = 0.45;
  const MIN_QUALITY = 0.52;
  const STORE_KEY = 'aieyescan.iris.v1';

  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = cropCanvas.height = CROP;
  const cropCtx = cropCanvas.getContext('2d', { willReadFrequently: true });

  const unwrapCanvas = document.createElement('canvas');
  unwrapCanvas.width = UW; unwrapCanvas.height = UR;
  const unwrapCtx = unwrapCanvas.getContext('2d');

  const PACKET = document.createElement('canvas'); // drawn into the visible unwrap view
  PACKET.width = 512; PACKET.height = 128;

  /* ---------------- Gabor bank ---------------- */
  const KERNELS = [];
  SCALES.forEach((scale, si) => {
    ORIENTS.forEach((deg, oi) => {
      const sigma = scale * 1.05;
      const lambda = scale * 3.2;
      const half = Math.ceil(3 * sigma);
      const size = half * 2 + 1;
      const re = new Float32Array(size * size), im = new Float32Array(size * size);
      const a = deg * Math.PI / 180, ca = Math.cos(a), sa = Math.sin(a);
      let energy = 0;
      for (let y = -half; y <= half; y++) {
        for (let x = -half; x <= half; x++) {
          const xr = x * ca + y * sa, yr = -x * sa + y * ca;
          const env = Math.exp(-(xr * xr + yr * yr) / (2 * sigma * sigma));
          const phase = 2 * Math.PI * xr / lambda;
          const i = (y + half) * size + (x + half);
          re[i] = env * Math.cos(phase);
          im[i] = env * Math.sin(phase);
          energy += env * env;
        }
      }
      const nrm = Math.sqrt(energy) || 1;
      for (let i = 0; i < re.length; i++) { re[i] /= nrm; im[i] /= nrm; }
      KERNELS.push({ si, oi, half, size, re, im });
    });
  });

  /* ---------------- pixel sampling ---------------- */
  function sampleRGB(d, w, h, x, y) {
    const x0 = Math.floor(x), y0 = Math.floor(y);
    const fx = x - x0, fy = y - y0;
    const x1 = Math.min(w - 1, x0 + 1), y1 = Math.min(h - 1, y0 + 1);
    const xa = clamp(x0, 0, w - 1), ya = clamp(y0, 0, h - 1);
    const i00 = (ya * w + xa) * 4, i10 = (ya * w + x1) * 4;
    const i01 = (y1 * w + xa) * 4, i11 = (y1 * w + x1) * 4;
    const w00 = (1 - fx) * (1 - fy), w10 = fx * (1 - fy);
    const w01 = (1 - fx) * fy, w11 = fx * fy;
    return [
      d[i00] * w00 + d[i10] * w10 + d[i01] * w01 + d[i11] * w11,
      d[i00 + 1] * w00 + d[i10 + 1] * w10 + d[i01 + 1] * w01 + d[i11 + 1] * w11,
      d[i00 + 2] * w00 + d[i10 + 2] * w10 + d[i01 + 2] * w01 + d[i11 + 2] * w11
    ];
  }

  /* ---------------- iris geometry from landmarks ---------------- */
  function geomFromIris(lms, centreIdx, contourIdx, Wv, Hv) {
    const c = lms[centreIdx];
    if (!c) return null;
    const pts = contourIdx.map(i => lms[i]).filter(Boolean);
    if (pts.length < 4) return null;
    let rSum = 0;
    for (const p of pts) rSum += Math.hypot((p.x - c.x) * Wv, (p.y - c.y) * Hv);
    const rp = rSum / pts.length;
    let ang = 0, best = 0;
    for (let i = 0; i < pts.length; i++) {
      for (let j = i + 1; j < pts.length; j++) {
        const d = Math.hypot((pts[j].x - pts[i].x) * Wv, (pts[j].y - pts[i].y) * Hv);
        if (d > best) {
          best = d;
          ang = Math.atan2((pts[j].y - pts[i].y) * Hv, (pts[j].x - pts[i].x) * Wv);
        }
      }
    }
    return { cx: c.x * Wv, cy: c.y * Hv, rp, ang, source: 'iris' };
  }

  // Fallback when refineLandmarks is unavailable: estimate from eye contour.
  function geomFromEye(lms, inner, outer, top, bottom, Wv, Hv) {
    const a = lms[inner], b = lms[outer], t = lms[top], d = lms[bottom];
    if (!a || !b || !t || !d) return null;
    const cx = (a.x + b.x) / 2 * Wv, cy = (a.y + b.y) / 2 * Hv;
    const eyeW = Math.hypot((b.x - a.x) * Wv, (b.y - a.y) * Hv);
    const eyeH = Math.hypot((d.x - t.x) * Wv, (d.y - t.y) * Hv);
    const rp = Math.max(6, Math.min(eyeW, eyeH) * 0.46);
    const ang = Math.atan2((b.y - a.y) * Hv, (b.x - a.x) * Wv);
    return { cx, cy, rp, ang, source: 'eye' };
  }

  function cropEye(video, g) {
    const scale = (CROP / 2) / Math.max(2, g.rp * PAD);
    cropCtx.setTransform(1, 0, 0, 1, 0, 0);
    cropCtx.fillStyle = '#02040a';
    cropCtx.fillRect(0, 0, CROP, CROP);
    cropCtx.save();
    cropCtx.translate(CROP / 2, CROP / 2);
    cropCtx.rotate(-g.ang);
    cropCtx.scale(scale, scale);
    cropCtx.translate(-g.cx, -g.cy);
    cropCtx.drawImage(video, 0, 0);
    cropCtx.restore();
    return { cx: CROP / 2, cy: CROP / 2, irisR: IRIS_R };
  }

  /* ---------------- pupil ---------------- */
  function findPupil(d, w, h, cx, cy, irisR) {
    const maxR = Math.round(irisR * 0.72);
    const prof = new Float32Array(maxR + 1);
    for (let r = 2; r <= maxR; r++) {
      let s = 0, n = 0;
      for (let a = 0; a < 72; a++) {
        const th = a / 72 * Math.PI * 2;
        const p = sampleRGB(d, w, h, cx + Math.cos(th) * r, cy + Math.sin(th) * r);
        s += 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]; n++;
      }
      prof[r] = s / n;
    }
    const sm = new Float32Array(maxR + 1);
    for (let r = 1; r <= maxR; r++) {
      let s = 0, n = 0;
      for (let k = -2; k <= 2; k++) {
        const j = r + k;
        if (j >= 1 && j <= maxR) { s += prof[j]; n++; }
      }
      sm[r] = s / n;
    }
    const lo = Math.max(3, Math.round(irisR * 0.10));
    const hi = Math.round(irisR * 0.66);
    let bestR = Math.round(irisR * 0.42), bestD = -1e9;
    for (let r = lo; r <= hi; r++) {
      const dv = sm[Math.min(maxR, r + 2)] - sm[Math.max(1, r - 2)];
      if (dv > bestD) { bestD = dv; bestR = r; }
    }
    const inner = sm[Math.max(1, Math.round(bestR * 0.35))];
    const outer = sm[Math.min(maxR, Math.round(bestR * 1.7))];
    const contrast = outer - inner;

    // circularity of the dark pupil region
    const thr = inner + Math.max(6, contrast * 0.5);
    const lim = Math.round(irisR * 0.85);
    const side = lim * 2 + 1;
    const dark = new Uint8Array(side * side);
    let area = 0, sx = 0, sy = 0;
    for (let y = -lim; y <= lim; y++) {
      for (let x = -lim; x <= lim; x++) {
        const px = Math.round(cx + x), py = Math.round(cy + y);
        if (px < 0 || py < 0 || px >= w || py >= h) continue;
        const i = (py * w + px) * 4;
        const L = 0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2];
        if (L < thr) { dark[(y + lim) * side + (x + lim)] = 1; area++; sx += x; sy += y; }
      }
    }
    let perimeter = 0;
    for (let y = 1; y < side - 1; y++) {
      for (let x = 1; x < side - 1; x++) {
        const i = y * side + x;
        if (!dark[i]) continue;
        if (!dark[i - 1] || !dark[i + 1] || !dark[i - side] || !dark[i + side]) perimeter++;
      }
    }
    const circ = perimeter > 8 ? clamp(4 * Math.PI * area / (perimeter * perimeter), 0, 1.2) : null;
    const offX = area ? sx / area : 0, offY = area ? sy / area : 0;
    const offset = Math.hypot(offX, offY) / Math.max(1, bestR);
    return {
      rp: bestR,
      circ,
      offset,
      contrast,
      darkFrac: area / (side * side)
    };
  }

  /* ---------------- unwrap ---------------- */
  function unwrapIris(d, w, h, cx, cy, pupilR, irisR) {
    const rgb = new Float32Array(UW * UR * 3);
    const lum = new Float32Array(UW * UR);
    const mask = new Uint8Array(UW * UR);
    const r0 = Math.max(2, pupilR * 1.12);
    const r1 = irisR * 0.97;
    let masked = 0, spec = 0;
    const hueSamples = [], satSamples = [], valSamples = [];
    for (let a = 0; a < UW; a++) {
      const th = a / UW * Math.PI * 2;
      const ca = Math.cos(th), sa = Math.sin(th);
      for (let r = 0; r < UR; r++) {
        const rr = r0 + (r1 - r0) * (r / (UR - 1));
        const p = sampleRGB(d, w, h, cx + ca * rr, cy + sa * rr);
        const i = a * UR + r;
        rgb[i * 3] = p[0]; rgb[i * 3 + 1] = p[1]; rgb[i * 3 + 2] = p[2];
        const L = 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2];
        lum[i] = L;
        const hsv = rgb2hsv(p[0], p[1], p[2]);
        const isSpec = hsv.v > 0.95 && hsv.s < 0.12;
        // crop the extreme inner/outer 6% (pupil fringe, limbal pigment ring)
        const edge = r < 2 || r > UR - 3;
        if (L < 18 || isSpec || edge) {
          mask[i] = 1; masked++;
          if (isSpec) spec++;
        } else if (r >= 4 && r <= Math.round(UR * 0.82)) {
          // colour is sampled away from the pupil fringe and the limbal ring,
          // which otherwise bias the iris colour towards dark brown.
          hueSamples.push(hsv.h);
          satSamples.push(hsv.s);
          valSamples.push(hsv.v);
        }
      }
    }
    const total = UW * UR;
    return {
      rgb, lum, mask,
      maskedFrac: masked / total,
      glareFrac: spec / total,
      hue: circMean(hueSamples),
      hueStd: circStd(hueSamples, circMean(hueSamples)),
      sat: mean(satSamples),
      val: mean(valSamples),
      valid: hueSamples.length
    };
  }

  /* ---------------- texture (crypts / pigment spots / pale spots) ---------------- */
  function blobAreas(cand, w, h) {
    const seen = new Uint8Array(w * h);
    const stack = new Int32Array(w * h);
    const areas = [];
    for (let i = 0; i < w * h; i++) {
      if (!cand[i] || seen[i]) continue;
      let sp = 0, area = 0;
      stack[sp++] = i; seen[i] = 1;
      while (sp) {
        const j = stack[--sp]; area++;
        const x = j % w, y = (j / w) | 0;
        const nb = [j - 1, j + 1, j - w, j + w];
        const ok = [x > 0, x < w - 1, y > 0, y < h - 1];
        for (let k = 0; k < 4; k++) {
          const q = nb[k];
          if (!ok[k] || seen[q] || !cand[q]) continue;
          seen[q] = 1; stack[sp++] = q;
        }
      }
      areas.push(area);
    }
    return areas;
  }

  function texture(d, w, h, cx, cy, pupilR, irisR) {
    const g = new Float32Array(w * h);
    for (let i = 0; i < w * h; i++) {
      g[i] = 0.299 * d[i * 4] + 0.587 * d[i * 4 + 1] + 0.114 * d[i * 4 + 2];
    }
    const iw = w + 1;
    const s = new Float64Array(iw * (h + 1)), s2 = new Float64Array(iw * (h + 1));
    for (let y = 0; y < h; y++) {
      let rs = 0, rs2 = 0;
      for (let x = 0; x < w; x++) {
        const v = g[y * w + x];
        rs += v; rs2 += v * v;
        s[(y + 1) * iw + (x + 1)] = s[y * iw + (x + 1)] + rs;
        s2[(y + 1) * iw + (x + 1)] = s2[y * iw + (x + 1)] + rs2;
      }
    }
    const win = 9, half = win >> 1;
    const lo = pupilR * 1.25, hi = irisR * 0.93;
    const dark = new Uint8Array(w * h), pale = new Uint8Array(w * h);
    for (let y = half; y < h - half; y++) {
      for (let x = half; x < w - half; x++) {
        const rr = Math.hypot(x - cx, y - cy);
        if (rr < lo || rr > hi) continue;
        const x0 = x - half, y0 = y - half, x1 = x + half + 1, y1 = y + half + 1;
        const sum = s[y1 * iw + x1] - s[y0 * iw + x1] - s[y1 * iw + x0] + s[y0 * iw + x0];
        const sum2 = s2[y1 * iw + x1] - s2[y0 * iw + x1] - s2[y1 * iw + x0] + s2[y0 * iw + x0];
        const n = win * win;
        const m = sum / n;
        const sd = Math.sqrt(Math.max(0, sum2 / n - m * m));
        const v = g[y * w + x];
        if (v < m - 1.15 * sd) dark[y * w + x] = 1;
        if (v > m + 1.15 * sd) pale[y * w + x] = 1;
      }
    }
    const darkAreas = blobAreas(dark, w, h);
    const paleAreas = blobAreas(pale, w, h);
    const annulus = Math.PI * (hi * hi - lo * lo);
    const cryptAreas = darkAreas.filter(a => a >= 6);
    return {
      crypts: cryptAreas.length,
      spots: darkAreas.filter(a => a >= 18).length,
      paleSpots: paleAreas.filter(a => a >= 10).length,
      cryptArea: cryptAreas.reduce((a, b) => a + b, 0) / Math.max(1, annulus)
    };
  }

  /* ---------------- iris code ---------------- */
  function buildCode(lum, mask) {
    const bits = new Uint8Array(CW * CH * PER);
    const bmask = new Uint8Array(CW * CH * PER);
    const tapCount = new Float32Array(KERNELS.length);
    KERNELS.forEach((K, i) => { tapCount[i] = (2 * K.half + 1) * (2 * K.half + 1); });

    // Remove angle-constant structure (concentric rings such as the collarette
    // and limbal ring) — those are shared by most irises. What varies with
    // angle is the individual fibre and crypt texture.
    const rowMean = new Float32Array(UR);
    for (let r = 0; r < UR; r++) {
      let s = 0;
      for (let a = 0; a < UW; a++) s += lum[a * UR + r];
      rowMean[r] = s / UW;
    }
    const base = new Float32Array(UW * UR);
    for (let a = 0; a < UW; a++) {
      for (let r = 0; r < UR; r++) base[a * UR + r] = lum[a * UR + r] - rowMean[r];
    }

    // Local contrast normalisation: remove shading and macro structure so the
    // phase code depends on iris texture rather than overall brightness.
    const norm = new Float32Array(UW * UR);
    const wa = 9, wr = 3;
    for (let a = 0; a < UW; a++) {
      for (let r = 0; r < UR; r++) {
        let s = 0, s2 = 0, n = 0;
        for (let da = -wa; da <= wa; da++) {
          const aa = ((a + da) % UW + UW) % UW;
          for (let dr = -wr; dr <= wr; dr++) {
            const rr = clamp(r + dr, 0, UR - 1);
            const v = base[aa * UR + rr];
            s += v; s2 += v * v; n++;
          }
        }
        const m = s / n;
        const sd = Math.sqrt(Math.max(0, s2 / n - m * m));
        norm[a * UR + r] = (base[a * UR + r] - m) / (sd + 6);
      }
    }

    for (let so = 0; so < KERNELS.length; so++) {
      const K = KERNELS[so], half = K.half, size = K.size;
      for (let gx = 0; gx < CW; gx++) {
        const ux = Math.round(gx * UW / CW);
        for (let gy = 0; gy < CH; gy++) {
          const uy = Math.round(gy * (UR - 1) / (CH - 1));
          let vr = 0, vi = 0, blocked = 0;
          for (let ky = -half; ky <= half; ky++) {
            const yy = uy + ky;
            if (yy < 0 || yy >= UR) continue;
            for (let kx = -half; kx <= half; kx++) {
              const xx = ((ux + kx) % UW + UW) % UW;
              const idx = xx * UR + yy;
              const kv = (ky + half) * size + (kx + half);
              vr += norm[idx] * K.re[kv];
              vi += norm[idx] * K.im[kv];
              if (mask[idx]) blocked++;
            }
          }
          const base = (gx * CH + gy) * PER + so * 2;
          bits[base] = vr >= 0 ? 1 : 0;
          bits[base + 1] = vi >= 0 ? 1 : 0;
          if (blocked > tapCount[so] * 0.30) bmask[base] = bmask[base + 1] = 1;
        }
      }
    }
    return { bits, bmask };
  }

  function hamming(a, ma, b, mb, maxShift) {
    let best = 1, bestShift = 0, bestRad = 0, bestN = 0;
    const inner = CH * PER;
    for (let s = -maxShift; s <= maxShift; s++) {
      for (let rs = -1; rs <= 1; rs++) {
        let diff = 0, n = 0;
        for (let gx = 0; gx < CW; gx++) {
          const bx = ((gx + s) % CW + CW) % CW;
          for (let gy = 0; gy < CH; gy++) {
            const by = gy + rs;
            if (by < 0 || by >= CH) continue;
            const baseA = (gx * CH + gy) * PER;
            const baseB = (bx * CH + by) * PER;
            for (let k = 0; k < PER; k++) {
              const ia = baseA + k, ib = baseB + k;
              // mask bit = 1 means the sample is occluded/invalid
              if (ma[ia] || mb[ib]) continue;
              n++; if (a[ia] !== b[ib]) diff++;
            }
          }
        }
        if (n > 0) {
          const d = diff / n;
          if (d < best) { best = d; bestShift = s; bestRad = rs; bestN = n; }
        }
      }
    }
    return { dist: best, shift: bestShift, radial: bestRad, compared: bestN, total: CW * CH * PER };
  }

  /* ---------------- bit packing / hashing ---------------- */
  function pack(bits) {
    const out = new Uint8Array(Math.ceil(bits.length / 8));
    for (let i = 0; i < bits.length; i++) if (bits[i]) out[i >> 3] |= 1 << (7 - (i & 7));
    return out;
  }
  function unpack(bytes, len) {
    const out = new Uint8Array(len);
    for (let i = 0; i < len; i++) out[i] = (bytes[i >> 3] >> (7 - (i & 7))) & 1;
    return out;
  }
  function toB64(bytes) {
    let s = '';
    for (let i = 0; i < bytes.length; i += 0x8000) {
      s += String.fromCharCode.apply(null, bytes.subarray(i, i + 0x8000));
    }
    return btoa(s);
  }
  function fromB64(str) {
    const raw = atob(str);
    const out = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) out[i] = raw.charCodeAt(i);
    return out;
  }
  async function signature(bytes) {
    try {
      if (window.crypto && crypto.subtle) {
        const digest = await crypto.subtle.digest('SHA-256', bytes);
        const u = new Uint8Array(digest).subarray(0, 8);
        return Array.from(u).map(x => x.toString(16).padStart(2, '0')).join('').toUpperCase()
          .replace(/(.{4})/g, '$1 ').trim();
      }
    } catch (e) { /* insecure context */ }
    let h = 0x811c9dc5;
    for (let i = 0; i < bytes.length; i++) { h ^= bytes[i]; h = Math.imul(h, 0x01000193) >>> 0; }
    return (h.toString(16).toUpperCase() + '00000000').slice(0, 8).replace(/(.{4})/g, '$1 ').trim();
  }

  /* ---------------- classification ---------------- */
  function classifyColour(uv) {
    const { h, s, v } = uv;
    if (v < 0.16) return { name: 'Very dark brown', code: 'brown' };
    if (s < 0.10) return { name: 'Grey', code: 'grey' };
    if (h >= 75 && h < 165) return { name: s > 0.35 ? 'Green' : 'Grey-green', code: 'green' };
    if (h >= 165 && h < 275) return { name: 'Blue', code: 'blue' };
    if (h >= 45 && h < 75) return { name: 'Hazel / amber', code: 'hazel' };
    if (h < 45) return { name: v < 0.32 ? 'Dark brown' : 'Brown', code: 'brown' };
    return { name: 'Mixed', code: 'mixed' };
  }

  function eyeColourSwatch(c) {
    switch (c.code) {
      case 'blue': return '#4f8fd6';
      case 'green': return '#5a9b6a';
      case 'grey': return '#8b98a8';
      case 'hazel': return '#a8833f';
      default: return '#6b4423';
    }
  }

  /* ---------------- one-eye analysis ---------------- */
  function bandLum(uv, a, b) {
    const r0 = Math.round(a * (UR - 1)), r1 = Math.round(b * (UR - 1));
    let s = 0, n = 0;
    for (let aa = 0; aa < UW; aa++) {
      for (let r = r0; r <= r1; r++) {
        const i = aa * UR + r;
        if (uv.mask[i]) continue;
        s += uv.lum[i]; n++;
      }
    }
    return n ? s / n : 0;
  }

  function analyseEye(d, w, h, info, geom, video, withCode = true) {
    const pupil = findPupil(d, w, h, info.cx, info.cy, info.irisR);
    const uv = unwrapIris(d, w, h, info.cx, info.cy, pupil.rp, info.irisR);
    const tex = texture(d, w, h, info.cx, info.cy, pupil.rp, info.irisR);
    const code = withCode ? buildCode(uv.lum, uv.mask) : null;
    const midLum = bandLum(uv, 0.42, 0.66);
    const outerLum = bandLum(uv, 0.82, 0.93);

    // focus: Laplacian variance inside the iris
    let lapSum = 0, lapSq = 0, lapN = 0;
    for (let y = 2; y < h - 2; y += 2) {
      for (let x = 2; x < w - 2; x += 2) {
        const rr = Math.hypot(x - info.cx, y - info.cy);
        if (rr < pupil.rp * 1.15 || rr > info.irisR * 0.92) continue;
        const i = y * w + x;
        const L = (a) => 0.299 * d[a * 4] + 0.587 * d[a * 4 + 1] + 0.114 * d[a * 4 + 2];
        const lap = Math.abs(4 * L(i) - L(i - 1) - L(i + 1) - L(i - w) - L(i + w));
        lapSum += lap; lapSq += lap * lap; lapN++;
      }
    }
    const lapMean = lapSum / Math.max(1, lapN);
    const lapVar = lapSq / Math.max(1, lapN) - lapMean * lapMean;

    const focus = clamp((lapVar - 25) / 320, 0, 1);
    const size = clamp((geom.rp - 8) / 22, 0, 1);
    const openness = clamp(1 - uv.maskedFrac * 2.4, 0, 1);
    const glareOK = clamp(1 - uv.glareFrac * 12, 0, 1);
    const pupilOK = pupil.contrast > 10 && pupil.rp > info.irisR * 0.12 ? 1 : 0.4;
    const quality = clamp(
      0.27 * focus + 0.24 * size + 0.22 * openness + 0.14 * glareOK + 0.13 * pupilOK,
      0, 1
    );

    const colour = classifyColour(uv);
    return {
      geom, uv, code, pupil, tex, focus, size, openness, glareOK, quality, colour,
      metrics: {
        irisPx: geom.rp,
        pupilRatio: pupil.rp / info.irisR,
        crypts: tex.crypts,
        cryptArea: tex.cryptArea,
        spots: tex.spots,
        paleSpots: tex.paleSpots,
        limbalRatio: midLum > 1 ? outerLum / midLum : null,
        hue: uv.hue,
        hueStd: uv.hueStd,
        sat: uv.sat,
        val: uv.val,
        maskedFrac: uv.maskedFrac,
        glareFrac: uv.glareFrac,
        pupilCirc: pupil.circ,
        pupilOffset: pupil.offset
      }
    };
  }

  /* ---------------- eye colour / appearance descriptions ---------------- */
  function colourMeta(m) {
    return `hue ${m.hue.toFixed(0)}° · saturation ${(m.sat * 100).toFixed(0)}% · brightness ${(m.val * 100).toFixed(0)}%`;
  }
  function heterochromia(a, b) {
    if (!a || !b) return { status: 'unknown', note: 'Both eyes must be clearly visible in the same frame.' };
    let dh = Math.abs(a.metrics.hue - b.metrics.hue); if (dh > 180) dh = 360 - dh;
    const dv = Math.abs(a.metrics.val - b.metrics.val);
    if (dh > 30 || dv > 0.25) {
      return {
        status: 'possible',
        note: `Left/right hue differ by ${dh.toFixed(0)}° and brightness by ${(dv * 100).toFixed(0)}%. Lighting and camera angle can cause this — confirm in identical lighting.`
      };
    }
    return { status: 'none', note: `Hue difference ${dh.toFixed(0)}° — within normal variation for a single iris pattern.` };
  }
  function anisocoria(a, b) {
    if (!a || !b) return { status: 'unknown', note: 'Both pupils must be visible in the same frame.' };
    const ra = a.metrics.pupilRatio, rb = b.metrics.pupilRatio;
    const diff = Math.abs(ra - rb) / Math.max(1e-6, (ra + rb) / 2);
    if (diff > 0.18) {
      return {
        status: 'possible',
        note: `Pupil sizes differ by ${(diff * 100).toFixed(0)}% in this photo. Anisocoria >0.4 mm is clinically relevant; this is not a measurement in millimetres and lighting is not controlled.`
      };
    }
    return { status: 'none', note: `Pupil-size difference ${(diff * 100).toFixed(0)}% — within the range explained by lighting and angle.` };
  }

  /* ---------------- ocular signs (screening flags only) ---------------- */
  function buildSigns(best, both) {
    const m = best.metrics;
    const signs = [];
    const add = (label, status, note) => signs.push({ label, status, note });

    // pigmented limbal ring (common, not a disease)
    if (m.limbalRatio != null) {
      const pct = Math.round((1 - m.limbalRatio) * 100);
      add('Limbal ring', 'observation',
        pct >= 20
          ? `The outer iris ring measures ${pct}% darker than the mid-iris — a prominent limbal ring, which is a normal variant and tends to become more visible with age.`
          : `The outer iris ring measures ${pct}% darker than the mid-iris, so no prominent limbal ring stands out.`);
    } else {
      add('Limbal ring', 'not assessable', 'The outer iris edge was too occluded to measure.');
    }

    // iris freckles / naevi
    if (m.spots >= 3) {
      add('Iris freckles / pigment spots', 'possible',
        `${m.spots} discrete pigment spots detected. Iris freckles are usually benign; an ophthalmologist should review any new, enlarging or multi-coloured lesion.`);
    } else {
      add('Iris freckles / pigment spots', 'none',
        `No discrete pigment spots above the detection threshold (${m.spots} found).`);
    }

    // irregular pupil margin
    if (m.pupilCirc !== null && m.pupilCirc < 0.78) {
      add('Pupil margin regularity', 'possible',
        `Pupil outline circularity ${m.pupilCirc.toFixed(2)} (a round pupil is ≈1.0). Irregularity can be caused by eyelashes, glare or a poor capture — or by synechiae/coloboma. Slit-lamp examination is required.`);
    } else {
      add('Pupil margin regularity', 'none',
        `Pupil outline is round within measurement error${m.pupilCirc !== null ? ' (' + m.pupilCirc.toFixed(2) + ')' : ''}.`);
    }

    // Kayser-Fleischer ring — a slit-lamp finding, never inferred from a photo
    add('Kayser–Fleischer ring (Wilson’s disease)', 'not assessable',
      'A KF ring is a copper deposit at the inner corneal edge, seen with a slit lamp under controlled illumination. A phone photo cannot detect or exclude it.');

    // Lisch nodules — explicitly not assessable
    add('Lisch nodules (neurofibromatosis type 1)', 'not assessable',
      'These are 1–2 mm dome-shaped nodules that require slit-lamp magnification. A phone photograph cannot detect or exclude them.');

    // Brushfield spots
    const lightIris = ['blue', 'green', 'grey'].includes(best.colour.code);
    if (lightIris) {
      if (m.paleSpots >= 5) {
        add('Brushfield spots', 'possible',
          `${m.paleSpots} pale spots were counted in a light iris. Brushfield spots occur in Down syndrome but are also common in healthy light-eyed people, especially infants — they are not diagnostic.`);
      } else {
        add('Brushfield spots', 'none',
          `No cluster of pale collarette spots reached the detection threshold (${m.paleSpots} pale spots found).`);
      }
    } else {
      add('Brushfield spots', 'not assessable',
        'Pale spots are only reliably visible in light irises; in a darker iris this photograph cannot assess them.');
    }

    // structural defects
    add('Iris coloboma', (m.pupilCirc !== null && m.pupilCirc < 0.60) ? 'possible' : 'none',
      (m.pupilCirc !== null && m.pupilCirc < 0.60)
        ? 'The pupil outline is markedly non-circular. A keyhole-shaped pupil can indicate a coloboma; ophthalmology review is required.'
        : 'No keyhole pupil defect was detected in this image.');

    add('Aniridia / absent iris', m.irisPx < 14 ? 'not assessable' : 'none',
      'A gross absence of iris tissue would be obvious; none is seen here. This is not a substitute for a clinical examination.');

    // anisocoria / heterochromia from both eyes
    const het = heterochromia(both.a, both.b);
    add('Heterochromia iridis', het.status, het.note);
    const ani = anisocoria(both.a, both.b);
    add('Anisocoria (unequal pupils)', ani.status, ani.note);

    // things people claim iridology can do
    add('Organ mapping / “iridology”', 'not supported',
      'Iridology — reading organs or systemic disease from iris zones — has repeatedly failed controlled evaluation and is not used in medicine. No output here is based on it.');

    return signs;
  }

  /* ---------------- status helpers ---------------- */
  function setStatus(title, sub, tone) {
    const t = $('irisIdStatus');
    if (t) {
      t.textContent = title;
      t.className = 'iris-status ' + (tone || '');
    }
    txt('irisIdSub', sub || '');
  }
  function setHint(msg, tone) {
    const e = $('irisHint');
    if (!e) return;
    e.textContent = msg;
    e.className = 'hint iris-hint ' + (tone || '');
  }
  function fmtPct(v) { return (clamp(v, 0, 1) * 100).toFixed(0) + '%'; }
  function setBar(id, v) {
    const e = $(id);
    if (e) e.style.width = (clamp(v, 0, 1) * 100).toFixed(1) + '%';
  }

  /* ---------------- scanner ---------------- */
  class IrisScanner {
    constructor() {
      this.video = $('irisVideo');
      this.overlay = $('irisOverlay');
      this.octx = this.overlay ? this.overlay.getContext('2d') : null;
      this.running = false;
      this.stream = null;
      this.mesh = null;
      this.modelErr = 0;
      this.lastLms = null;
      this.lastSend = 0;
      this.lastMeasure = 0;
      this.busy = false;
      this.raf = 0;
      this.geoms = [];
      this.last = null;
      this.both = { a: null, b: null };
      this.template = null;
      this.frames = 0; this.fps = 0; this.fpsT = 0;
      this.bind();
      this.loadTemplate();
    }

    bind() {
      const start = $('irisStart');
      if (start) start.addEventListener('click', () => this.start());
      const stop = $('irisStop');
      if (stop) stop.addEventListener('click', () => this.stop(true));
      const an = $('irisAnalyze');
      if (an) an.addEventListener('click', () => this.analyseNow());
      const en = $('irisEnroll');
      if (en) en.addEventListener('click', () => this.enroll());
      const ve = $('irisVerify');
      if (ve) ve.addEventListener('click', () => this.verify());
      const fo = $('irisForget');
      if (fo) fo.addEventListener('click', () => this.forget());
      const consent = $('irisConsent');
      if (consent) consent.addEventListener('change', () => this.refreshButtons());
    }

    /* ---------- template storage ---------- */
    loadTemplate() {
      try {
        const raw = localStorage.getItem(STORE_KEY);
        if (raw) {
          const t = JSON.parse(raw);
          if (t && t.code) {
            if (t.w === CW && t.h === CH && t.per === PER) {
              this.template = t;
            } else {
              // Code geometry changed between builds — the old template is not comparable.
              localStorage.removeItem(STORE_KEY);
              setStatus('Stored ID is from an older build', 'Its iris code layout no longer matches. Enrol again to create a new local ID.', 'warn');
            }
          }
        }
      } catch (e) { this.template = null; }
      this.refreshButtons();
      if (this.template) {
        setStatus('Local ID on this device',
          `Created ${new Date(this.template.created).toLocaleDateString()}. Turn on the camera and press Verify.`, 'idle');
      }
    }
    saveTemplate(t) {
      try { localStorage.setItem(STORE_KEY, JSON.stringify(t)); } catch (e) { /* storage blocked */ }
      this.template = t;
      this.refreshButtons();
    }
    forget() {
      try { localStorage.removeItem(STORE_KEY); } catch (e) { /* noop */ }
      this.template = null;
      this.refreshButtons();
      setStatus('Local ID removed', 'The stored iris template has been deleted from this browser.', 'idle');
      txt('irisMatchDist', '—');
      const bar = $('irisMatchBar'); if (bar) { bar.style.width = '0%'; bar.style.background = 'var(--cyan)'; }
    }
    refreshButtons() {
      const has = !!this.template;
      const ve = $('irisVerify'); if (ve) ve.disabled = !has || !this.running;
      const fo = $('irisForget'); if (fo) fo.disabled = !has;
      const en = $('irisEnroll');
      if (en) {
        const c = $('irisConsent');
        en.disabled = !(c && c.checked) || !this.running;
      }
      const an = $('irisAnalyze');
      if (an) an.disabled = !this.running;
    }

    /* ---------- camera ---------- */
    async start() {
      if (this.running) return;
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        setHint('This browser does not expose the camera API. Use Chrome, Edge or Firefox on https/localhost.', 'bad');
        return;
      }
      // one camera at a time: stop the rPPG scan first
      if (window.RPPG && window.RPPG.running) window.RPPG.stop(false);

      setHint('Requesting camera…', '');
      try {
        this.stream = await navigator.mediaDevices.getUserMedia({
          video: { width: { ideal: 1920 }, height: { ideal: 1080 }, facingMode: 'user' },
          audio: false
        });
      } catch (e) {
        setHint('Camera permission denied. Allow it from the camera icon in the address bar, then retry.', 'bad');
        return;
      }
      this.video.srcObject = this.stream;
      try { await this.video.play(); } catch (e) { /* autoplay guard */ }
      const empty = $('irisEmpty'); if (empty) empty.style.display = 'none';
      this.running = true;
      this.frames = 0; this.fpsT = performance.now();
      this.refreshButtons();
      setHint('Camera live — bring one eye close to the lens, open it wide and hold still.', '');

      try {
        this.mesh = await window.FaceMeshShared.get();
        if (!this.unsub) {
          this.unsub = window.FaceMeshShared.subscribe(r => {
            this.lastLms = (r.multiFaceLandmarks && r.multiFaceLandmarks[0]) || null;
          });
        }
      } catch (e) {
        this.mesh = null;
        setHint('Face-mesh model could not start. Reload the page; if it persists, check that vendor/face_mesh/ was deployed.', 'bad');
      }

      this.sizeCanvases();
      this.loop();
    }

    stop(user) {
      this.running = false;
      cancelAnimationFrame(this.raf);
      if (this.stream) { this.stream.getTracks().forEach(t => t.stop()); this.stream = null; }
      if (this.video) this.video.srcObject = null;
      this.lastLms = null; this.geoms = []; this.last = null;
      this.both = { a: null, b: null };
      const empty = $('irisEmpty');
      if (empty) {
        empty.style.display = '';
        const h = $('irisEmptyTitle'); if (h) h.textContent = 'Camera stopped';
        const p = $('irisEmptyText'); if (p) p.textContent = 'Start again whenever you are ready.';
      }
      const b = $('irisBadgeEyes'); if (b) { b.textContent = 'Eyes: —'; b.className = 'badge'; }
      const q = $('irisBadgeQuality'); if (q) { q.textContent = 'Quality: —'; q.className = 'badge'; }
      const f = $('irisBadgeFps'); if (f) f.textContent = '— fps';
      if (this.octx && this.overlay) this.octx.clearRect(0, 0, this.overlay.width, this.overlay.height);
      setHint(user ? 'Camera stopped. Start again whenever you are ready.' : 'Camera released.', '');
      this.refreshButtons();
    }

    sizeCanvases() {
      if (!this.video || !this.overlay) return;
      if (this.video.videoWidth) {
        this.overlay.width = this.video.videoWidth;
        this.overlay.height = this.video.videoHeight;
      }
    }

    /* ---------- loop ---------- */
    async loop() {
      if (!this.running) return;
      this.raf = requestAnimationFrame(() => this.loop());
      const now = performance.now();

      if (this.mesh && this.video.readyState >= 2 && now - this.lastSend > 66 && !this.busy) {
        this.busy = true;
        try {
          await window.FaceMeshShared.send(this.mesh, this.video);
          this.modelErr = 0;
        } catch (e) {
          if (++this.modelErr > 30) {
            this.mesh = null;
            setHint('Face-mesh inference is failing. Close other tabs using the camera and restart the scan.', 'bad');
          }
        }
        this.busy = false;
        this.lastSend = now;
        this.sizeCanvases();
      }

      this.frames++;
      if (now - this.fpsT > 1000) {
        this.fps = this.frames * 1000 / (now - this.fpsT);
        this.frames = 0; this.fpsT = now;
        txt('irisBadgeFps', this.fps.toFixed(0) + ' fps');
      }

      // heavy measurements ~3x/second
      if (now - this.lastMeasure > 320) {
        this.lastMeasure = now;
        this.measure();
      }
      this.draw();
    }

    /* ---------- landmarks -> measurements ---------- */
    measure() {
      const v = this.video;
      if (!v || !v.videoWidth) return;
      const Wv = v.videoWidth, Hv = v.videoHeight;
      const l = this.lastLms;

      if (!l) {
        this.geoms = [];
        this.last = null;
        const b = $('irisBadgeEyes');
        if (b) { b.textContent = 'Eyes: not found'; b.className = 'badge bad'; }
        setHint('No face detected — face the camera in even light and bring one eye close to the lens.', 'warn');
        return;
      }

      let geoms = [];
      if (l.length >= 478) {
        const g1 = geomFromIris(l, 468, [469, 470, 471, 472], Wv, Hv);
        const g2 = geomFromIris(l, 473, [474, 475, 476, 477], Wv, Hv);
        if (g1) geoms.push(g1);
        if (g2) geoms.push(g2);
      }
      if (!geoms.length) {
        const g1 = geomFromEye(l, 33, 133, 159, 145, Wv, Hv);
        const g2 = geomFromEye(l, 362, 263, 386, 374, Wv, Hv);
        if (g1) geoms.push(g1);
        if (g2) geoms.push(g2);
      }
      geoms.sort((a, b) => b.rp - a.rp);
      this.geoms = geoms;

      if (!geoms.length || geoms[0].rp < 7) {
        this.last = null;
        const b = $('irisBadgeEyes');
        if (b) { b.textContent = 'Eyes: too small'; b.className = 'badge warn'; }
        setHint('Iris too small in frame — move closer until the eye fills most of the view.', 'warn');
        return;
      }

      // analyse up to two eyes
      const results = [];
      for (let i = 0; i < Math.min(2, geoms.length); i++) {
        const g = geoms[i];
        const info = cropEye(v, g);
        let d;
        try { d = cropCtx.getImageData(0, 0, CROP, CROP); }
        catch (e) { continue; }
        results.push(analyseEye(d.data, CROP, CROP, info, g, v, false));
      }
      if (!results.length) return;
      results.sort((a, b) => b.quality - a.quality);
      // Only the best eye gets the expensive Gabor code; the other contributes
      // colour/pupil metrics for heterochromia and anisocoria.
      results[0].code = buildCode(results[0].uv.lum, results[0].uv.mask);
      this.last = results[0];
      this.both = { a: results[0], b: results[1] || null };

      // live UI
      const b = $('irisBadgeEyes');
      if (b) {
        b.textContent = `Eyes: ${results.length} · iris ${Math.round(results[0].geom.rp)} px`;
        b.className = 'badge ' + (results[0].geom.rp >= 16 ? 'ok' : 'warn');
      }
      const q = $('irisBadgeQuality');
      if (q) {
        const pct = Math.round(results[0].quality * 100);
        q.textContent = `Quality: ${pct}%`;
        q.className = 'badge ' + (results[0].quality >= MIN_QUALITY ? 'ok' : pct >= 40 ? 'warn' : 'bad');
      }
      const best = results[0];
      setBar('qFocus', best.focus);
      setBar('qSize', best.size);
      setBar('qOpen', best.openness);
      setBar('qGlare', best.glareOK);
      txt('qFocusVal', fmtPct(best.focus));
      txt('qSizeVal', fmtPct(best.size));
      txt('qOpenVal', fmtPct(best.openness));
      txt('qGlareVal', fmtPct(best.glareOK));

      const sw = $('irisLiveSwatch');
      if (sw) sw.style.background = eyeColourSwatch(best.colour);
      txt('irisLiveColour', best.colour.name);

      if (best.quality >= MIN_QUALITY) {
        setHint('Good capture — press “Analyse iris” or create a local ID.', 'ok');
      } else if (best.focus < 0.35) {
        setHint('Image is soft — hold still, tap the screen to focus, and avoid digital zoom.', 'warn');
      } else if (best.size < 0.4) {
        setHint('Move closer — the iris should be at least ~18 px across in the camera frame.', 'warn');
      } else if (best.openness < 0.5) {
        setHint('Part of the iris is covered — open the eye wider and avoid eyelashes.', 'warn');
      } else {
        setHint('Glare on the eye — turn away from the window/lamp or tilt the phone slightly.', 'warn');
      }

      this.paintUnwrap(best.uv);
    }

    paintUnwrap(uv) {
      const target = $('irisUnwrap');
      if (!target) return;
      const img = unwrapCtx.createImageData(UW, UR);
      for (let i = 0; i < UW * UR; i++) {
        img.data[i * 4] = uv.rgb[i * 3];
        img.data[i * 4 + 1] = uv.rgb[i * 3 + 1];
        img.data[i * 4 + 2] = uv.rgb[i * 3 + 2];
        img.data[i * 4 + 3] = uv.mask[i] ? 90 : 255;
      }
      unwrapCtx.putImageData(img, 0, 0);
      const c = target.getContext('2d');
      c.imageSmoothingEnabled = true;
      c.clearRect(0, 0, target.width, target.height);
      c.drawImage(unwrapCanvas, 0, 0, target.width, target.height);
    }

    /* ---------- overlay ---------- */
    draw() {
      const c = this.octx, o = this.overlay;
      if (!c || !o || !o.width) return;
      c.clearRect(0, 0, o.width, o.height);
      const good = this.last && this.last.quality >= MIN_QUALITY;
      const colour = good ? 'rgba(52,226,208,.95)' : 'rgba(251,191,36,.9)';
      this.geoms.forEach((g, idx) => {
        c.strokeStyle = idx === 0 ? colour : 'rgba(139,124,246,.65)';
        c.lineWidth = Math.max(2, o.width / 420);
        c.beginPath();
        c.arc(g.cx, g.cy, g.rp * 1.06, 0, Math.PI * 2);
        c.stroke();
        // crosshair ticks
        c.strokeStyle = 'rgba(232,238,255,.5)';
        c.lineWidth = 1;
        const t = g.rp * 0.22;
        c.beginPath();
        c.moveTo(g.cx - g.rp - t, g.cy); c.lineTo(g.cx - g.rp + t, g.cy);
        c.moveTo(g.cx + g.rp - t, g.cy); c.lineTo(g.cx + g.rp + t, g.cy);
        c.moveTo(g.cx, g.cy - g.rp - t); c.lineTo(g.cx, g.cy - g.rp + t);
        c.moveTo(g.cx, g.cy + g.rp - t); c.lineTo(g.cx, g.cy + g.rp + t);
        c.stroke();
      });
      // pupil of the eye actually being analysed
      if (this.last && this.last.geom) {
        const g = this.last.geom;
        c.strokeStyle = 'rgba(56,189,248,.85)';
        c.lineWidth = Math.max(1.5, o.width / 720);
        c.beginPath();
        c.arc(g.cx, g.cy, this.last.pupil.rp * (g.rp / IRIS_R), 0, Math.PI * 2);
        c.stroke();
      }
      // corner brackets
      c.strokeStyle = 'rgba(56,189,248,.6)';
      c.lineWidth = Math.max(2, o.width / 500);
      const p = o.width * 0.015, len = o.width * 0.04;
      [[p, p, 1, 1], [o.width - p, p, -1, 1], [p, o.height - p, 1, -1], [o.width - p, o.height - p, -1, -1]].forEach(([x, y, sx, sy]) => {
        c.beginPath();
        c.moveTo(x + sx * len, y); c.lineTo(x, y); c.lineTo(x, y + sy * len);
        c.stroke();
      });
    }

    /* ---------- full analysis ---------- */
    async analyseNow() {
      const best = this.last;
      if (!best) {
        setHint('Nothing to analyse yet — no iris captured.', 'warn');
        return;
      }
      if (best.quality < MIN_QUALITY) {
        setHint(`Capture quality ${fmtPct(best.quality)} is too low for a reliable read. Improve the capture and try again.`, 'warn');
        return;
      }
      const sig = await signature(pack(best.code.bits));
      best.signature = sig;
      this.render(best, this.both);
    }

    async render(best, both) {
      const m = best.metrics;
      const sig = best.signature || await signature(pack(best.code.bits));
      best.signature = sig;
      txt('irisSignature', sig);
      const enrolled = !!this.template;
      setStatus(
        enrolled ? 'Signature captured · local ID on device' : 'Signature captured',
        enrolled
          ? 'Computed locally. Press Verify to compare this capture with the stored template.'
          : 'Computed locally from the iris texture. Not enrolled — tick the consent box and create a local ID to verify a later scan.',
        'ok'
      );

      // colour
      txt('irisColourName', best.colour.name);
      txt('irisColourMeta', colourMeta(m));
      const sw = $('irisSwatch');
      if (sw) sw.style.background = eyeColourSwatch(best.colour);

      // phenotype
      txt('irisPupil', (m.pupilRatio * 100).toFixed(0) + '%');
      txt('irisPupilMeta', `pupil diameter ÷ iris diameter · estimated from this photo, not a pupillometer`);
      txt('irisCrypts', m.crypts);
      txt('irisCryptsMeta', `crypt density ${(m.cryptArea * 100).toFixed(1)}% of the iris annulus`);
      txt('irisSpots', m.spots);
      txt('irisSpotsMeta', `${m.spots} pigment spots above the size threshold`);
      txt('irisTexture', `${(m.cryptArea * 100).toFixed(1)}%`);
      txt('irisTextureMeta', `masked pixels ${(m.maskedFrac * 100).toFixed(0)}% · glare ${(m.glareFrac * 100).toFixed(1)}%`);
      txt('irisHueMeta', `hue spread ±${m.hueStd.toFixed(0)}°`);

      const het = heterochromia(both.a, both.b);
      txt('irisHetero', het.status === 'possible' ? 'Possible' : het.status === 'none' ? 'Not detected' : 'Not measurable');
      txt('irisHeteroMeta', het.note);

      const ani = anisocoria(both.a, both.b);
      txt('irisAniso', ani.status === 'possible' ? 'Possible' : ani.status === 'none' ? 'Not detected' : 'Not measurable');
      txt('irisAnisoMeta', ani.note);

      // signs
      const ul = $('irisSigns');
      if (ul) {
        ul.innerHTML = '';
        buildSigns(best, both).forEach(s => {
          const li = document.createElement('li');
          li.className = 'iris-sign ' + String(s.status).replace(/\s+/g, '-');
          const head = document.createElement('div');
          head.className = 'iris-sign-head';
          const b = document.createElement('b');
          b.textContent = s.label;
          const tag = document.createElement('span');
          tag.className = 'tag ' + (s.status === 'possible' ? 'mod' : s.status === 'none' ? 'low' : s.status === 'not supported' ? 'high' : '');
          tag.textContent = ({
            possible: 'Possible',
            none: 'Not detected',
            'not assessable': 'Not assessable',
            observation: 'Observation',
            unknown: 'Not measurable',
            'not supported': 'Not supported'
          })[s.status] || s.status;
          head.appendChild(b); head.appendChild(tag);
          const p = document.createElement('p');
          p.textContent = s.note;
          li.appendChild(head); li.appendChild(p);
          ul.appendChild(li);
        });
      }

      // scroll results into view
      const res = document.querySelector('.iris-results');
      if (res) {
        const r = res.getBoundingClientRect();
        if (r.top > window.innerHeight - 90 || r.bottom < 100) {
          try { res.scrollIntoView({ behavior: 'smooth', block: 'center' }); } catch (e) { /* noop */ }
        }
      }
    }

    /* ---------- enrol / verify ---------- */
    enroll() {
      const best = this.last;
      const consent = $('irisConsent');
      if (!best) { setHint('No iris capture yet — start the camera and hold still.', 'warn'); return; }
      if (!consent || !consent.checked) {
        setHint('Tick the storage consent box first — an iris template is sensitive biometric data.', 'warn');
        return;
      }
      if (best.quality < MIN_QUALITY) {
        setHint(`Quality ${fmtPct(best.quality)} is too low to enrol. Improve lighting and distance, then retry.`, 'warn');
        return;
      }
      const t = {
        v: 1,
        created: Date.now(),
        w: CW, h: CH, per: PER,
        code: toB64(pack(best.code.bits)),
        mask: toB64(pack(best.code.bmask))
      };
      this.saveTemplate(t);
      setStatus('Local ID created',
        'Template stored in this browser only (localStorage). It never leaves this device. Anyone with access to this browser profile can read it — use Forget when done.',
        'ok');
      setHint('Enrolled. Look away, then look back and press Verify to test a genuine match.', 'ok');
    }

    verify() {
      const best = this.last;
      if (!this.template) { setHint('No local ID stored yet.', 'warn'); return; }
      if (!best) { setHint('No live capture yet.', 'warn'); return; }
      if (best.quality < MIN_QUALITY) {
        setHint(`Live capture quality ${fmtPct(best.quality)} is too low to verify. Improve the capture.`, 'warn');
        return;
      }
      const code = unpack(fromB64(this.template.code), CW * CH * PER);
      const mask = unpack(fromB64(this.template.mask), CW * CH * PER);
      const r = hamming(best.code.bits, best.code.bmask, code, mask, 4);
      const dist = r.dist;
      const comparedPct = r.compared / r.total;
      txt('irisMatchDist', dist.toFixed(3) + (r.shift ? ` · rotation ${r.shift * (360 / CW)}°` : ''));
      const bar = $('irisMatchBar');
      if (bar) {
        bar.style.width = clamp(dist * 100, 0, 100) + '%';
        bar.style.background = dist <= GENUINE_MAX ? 'var(--cyan)'
          : dist <= INCONCLUSIVE_MAX ? 'var(--amber)' : 'var(--danger)';
      }

      if (comparedPct < 0.25) {
        setStatus('Inconclusive', `Only ${fmtPct(comparedPct)} of the code was usable — recapture with less eyelid/glare.`, 'warn');
        return;
      }
      if (dist <= GENUINE_MAX) {
        setStatus('Genuine match', `Hamming distance ${dist.toFixed(3)} (threshold ${GENUINE_MAX}). Compared ${fmtPct(comparedPct)} of the code.`, 'ok');
      } else if (dist <= INCONCLUSIVE_MAX) {
        setStatus('Inconclusive', `Hamming distance ${dist.toFixed(3)} sits between the genuine (${GENUINE_MAX}) and impostor (${INCONCLUSIVE_MAX}) thresholds. Recapture and try again.`, 'warn');
      } else {
        setStatus('No match', `Hamming distance ${dist.toFixed(3)} — more like a different iris than the stored template.`, 'bad');
      }
    }
  }

  /* ---------------- boot ---------------- */
  if (document.getElementById('irisStage')) {
    window.Iris = new IrisScanner();
  }

  /* Test hook: run the image pipeline on a caller-supplied canvas.
     Used by the QA harness; not referenced by the UI. */
  window.IrisDebug = {
    analyseCanvas(canvas) {
      const ctx = canvas.getContext('2d', { willReadFrequently: true });
      const w = canvas.width, h = canvas.height;
      const d = ctx.getImageData(0, 0, w, h);
      const irisR = Math.min(w, h) * 0.36;
      const info = { cx: w / 2, cy: h / 2, irisR };
      const geom = { cx: w / 2, cy: h / 2, rp: irisR, ang: 0, source: 'test' };
      return analyseEye(d.data, w, h, info, geom, null);
    },
    hamming, pack, unpack, buildSigns, colourMeta, CW, CH, PER
  };
})();
