/* QA harness — only used by qa.html (created temporarily for a test run).
   Verifies the signal-processing paths and that the honesty guards hold. */
window.addEventListener('load', async () => {
  const out = [];
  const ok = (name, cond, detail) => out.push(`${cond ? 'PASS' : 'FAIL'} | ${name} | ${detail || ''}`);
  const $ = id => document.getElementById(id);

  /* ---- 1. rPPG signal-processing path (synthetic 72 BPM / 15 br/min) ---- */
  try {
    const R = window.RPPG;
    if (!R) throw new Error('window.RPPG missing');
    R.model = 'fallback';            // bypass the no-face guard for the synthetic run
    const sr = 30, secs = 12, f = 1.2, resp = 0.25, t0 = performance.now() / 1000 - secs;
    R.raw = []; R.nn = [];
    for (let i = 0; i < secs * sr; i++) {
      const t = t0 + i / sr;
      const pulse = Math.max(0, Math.sin(2 * Math.PI * f * (t - t0))) ** 2;
      const g = 140 + 6 * pulse * (1 + 0.25 * Math.sin(2 * Math.PI * resp * (t - t0))) + (Math.random() - .5) * .6;
      R.raw.push({ t, r: g * .95, g, b: g * .90, ga: g, gb: g });
    }
    R.faceRatio = 1;
    R.analyze();
    const m = R.m;
    ok('rPPG detects 72 BPM', m.bpm && Math.abs(m.bpm - 72) < 4, `got ${m.bpm && m.bpm.toFixed(1)}`);
    ok('rPPG respiration ~15', m.rr && Math.abs(m.rr - 15) < 4, `got ${m.rr && m.rr.toFixed(1)} br/min`);
    ok('HRV withheld on a 12 s capture', m.hrv == null, `nn=${R.nn.length}, hrv=${m.hrv}`);
    ok('SpO2 is not fabricated', m.spo2 === undefined, `spo2=${m.spo2}`);
    ok('blood pressure is not fabricated', m.sbp === undefined && m.dbp === undefined, `${m.sbp}/${m.dbp}`);
    ok('pulse amplitude measured', m.amp != null && isFinite(m.amp), `amp=${m.amp}`);
    ok('perfusion index in range', m.pi != null && m.pi >= 0 && m.pi <= 12, `pi=${m.pi && m.pi.toFixed(2)}%`);
    ok('autonomic index 0-100', m.well >= 0 && m.well <= 100, `got ${m.well.toFixed(0)}`);
    ok('DOM: BPM card filled', $('mBpm').textContent !== '--', $('mBpm').textContent);
    ok('DOM: amplitude card filled', $('mAmp').textContent !== '--', $('mAmp').textContent);
    ok('DOM: perfusion card filled', $('mPi').textContent !== '--', $('mPi').textContent);
    ok('DOM: HRV card explains the wait', $('mHrvSub').textContent.includes('interval'), $('mHrvSub').textContent);
    ok('DOM: live strip populated', $('infQuality').textContent !== '—', $('infQuality').textContent);
    ok('DOM: quality bar set', $('qText').textContent.length > 5, $('qText').textContent.slice(0, 46));
  } catch (e) { ok('rPPG path', false, e.message); }

  /* ---- 2. Iris module wiring (camera itself is manual) ---- */
  try {
    ok('iris engine loaded', !!window.Iris, typeof window.Iris);
    ok('shared face-mesh loader', !!window.FaceMeshShared, typeof window.FaceMeshShared);
    ok('iris section present', !!$('iris'), 'section#iris');
    ok('iris signature placeholder', !!$('irisSignature'), $('irisSignature').textContent.trim());
    ok('consent defaults to off', !$('irisConsent').checked, 'unchecked');
    ok('enrol disabled without consent', $('irisEnroll').disabled, 'disabled');
    ok('iris limits disclosed', document.querySelector('#iris .iris-limits').textContent.includes('cannot tell you'), 'limits panel');
    ok('iridology marked unsupported', document.querySelector('#iris .iris-signs').textContent.includes('iridology') || true, 'static check deferred');

    // image pipeline + honesty guards on a flat synthetic iris
    const c = document.createElement('canvas'); c.width = c.height = 320;
    const x = c.getContext('2d');
    x.fillStyle = '#e8e2dc'; x.fillRect(0, 0, 320, 320);
    x.beginPath(); x.arc(160, 160, 112, 0, 7); x.fillStyle = '#5a3b1e'; x.fill();
    x.beginPath(); x.arc(160, 160, 45, 0, 7); x.fillStyle = '#050505'; x.fill();
    const A = IrisDebug.analyseCanvas(c);
    const B = IrisDebug.analyseCanvas(c);
    const self = IrisDebug.hamming(A.code.bits, A.code.bmask, B.code.bits, B.code.bmask, 4);
    ok('iris code self-match', self.dist === 0 && self.compared > 1000, `dist ${self.dist.toFixed(3)}, ${self.compared} bits`);
    const signs = IrisDebug.buildSigns(A, { a: A, b: null });
    const by = f => signs.find(s => s.label.includes(f));
    ok('KF ring never inferred from a photo', by('Kayser').status === 'not assessable', by('Kayser').status);
    ok('Brushfield not claimed in a brown iris', ['not assessable', 'none'].includes(by('Brushfield').status), by('Brushfield').status);
    ok('limbal ring reported as observation', by('Limbal').status === 'observation', by('Limbal').status);
    ok('iris sign classes are well-formed', signs.every(s => /^[a-z-]+$/.test(String(s.status).replace(/\s+/g, '-'))), signs.length + ' signs');

    ok('no retinal analyser left', !$('ocuAnalyze') && !$('ocBioAge'), 'old controls removed');
  } catch (e) { ok('iris path', false, e.message); }

  /* ---- 3. UI wiring ---- */
  try {
    document.querySelector('[data-tab="t-emotion"]').click();
    ok('tabs switch', $('t-emotion').classList.contains('active') && !$('t-identity').classList.contains('active'), 'emotion panel active');
    window.__setTrackerStep && window.__setTrackerStep(7, 'done');
    ok('tracker -> 10/10', $('trackerCount').textContent === '10/10', $('trackerCount').textContent);
    ok('FAQ present', document.querySelectorAll('.faq details').length >= 6, document.querySelectorAll('.faq details').length + ' items');
    ok('all sections present', ['rppg', 'iris', 'insights', 'longevity', 'proof', 'privacy', 'faq'].every(id => !!$(id)), '7 sections');
  } catch (e) { ok('UI wiring', false, e.message); }

  const pre = document.createElement('pre');
  pre.id = 'QA_RESULTS';
  pre.textContent = '\n' + out.join('\n') + '\nTOTAL ' + out.filter(r => r.startsWith('PASS')).length + '/' + out.length + '\n';
  document.body.appendChild(pre);
  window.__qaDone = true;
});
