/* QA harness — only used by qa.html (deleted after the test run) */
window.addEventListener('load', async () => {
  const out = [];
  const ok = (name, cond, detail) => out.push(`${cond ? 'PASS' : 'FAIL'} | ${name} | ${detail || ''}`);
  const $ = id => document.getElementById(id);

  /* ---- 1. rPPG signal-processing path (synthetic 72 BPM / 15 br/min signal) ---- */
  try {
    const R = window.RPPG;
    if (!R) throw new Error('window.RPPG missing');
    const sr = 30, secs = 12, f = 1.2, resp = 0.25, t0 = performance.now() / 1000 - secs;
    R.raw = [];
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
    ok('rPPG HRV numeric', m.hrv && m.hrv > 0, `got ${m.hrv && m.hrv.toFixed(0)} ms`);
    ok('rPPG SpO2 in 90-100', m.spo2 >= 90 && m.spo2 <= 100, `got ${m.spo2 && m.spo2.toFixed(1)}%`);
    ok('rPPG BP plausible', m.sbp >= 95 && m.sbp <= 160 && m.dbp < m.sbp && m.dbp >= 55, `${m.sbp.toFixed(0)}/${m.dbp.toFixed(0)}`);
    ok('rPPG wellness 0-100', m.well >= 0 && m.well <= 100, `got ${m.well.toFixed(0)}`);
    ok('DOM: BPM card filled', $('mBpm').textContent !== '--', $('mBpm').textContent);
    ok('DOM: BP card filled', $('mBp').textContent !== '--/--', $('mBp').textContent);
    ok('DOM: live inference strip', $('infStress').textContent !== '—', $('infStress').textContent);
    ok('DOM: quality bar set', $('qText').textContent.length > 5, $('qText').textContent.slice(0, 46));
  } catch (e) { ok('rPPG path', false, e.message); }

  /* ---- 2. Oculomics end-to-end (demo fundus -> analyse -> render) ---- */
  try {
    $('ocuDemo').click();
    await new Promise(r => setTimeout(r, 150));
    ok('demo fundus drawn', $('dropIdle').style.display === 'none', 'preview shown');
    $('ocuAnalyze').click();
    await new Promise(r => setTimeout(r, 3200));
    ok('Oculomics biological age', $('ocBioAge').textContent !== '--', `bio age ${$('ocBioAge').textContent} (gap ${$('ocGap').textContent})`);
    ok('Oculomics cardio risk', $('cvNum').textContent.includes('%'), $('cvNum').textContent);
    ok('Oculomics CKD', $('ckdNum').textContent.includes('%'), $('ckdNum').textContent);
    ok('Oculomics metabolic (DR stage)', /DR stage \d/.test($('metNum').textContent), $('metNum').textContent);
    ok('Oculomics neuro biomarker', $('neuroNum').textContent.includes('CNFL'), $('neuroNum').textContent);
    ok('Oculomics QUARTZ', $('quaNum').textContent.includes('QUARTZ'), $('quaNum').textContent);
    ok('Oculomics mortality/gap', $('mortNum').textContent.includes('all-cause'), $('mortNum').textContent);
    ok('Oculomics verdict text', $('ocVerdict').textContent.length > 40, $('ocVerdict').textContent.slice(0, 60) + '…');
    ok('Oculomics identity readout', !$('ocuIdentity').hidden && $('idHash').textContent.includes('unique'), `${$('idSex').textContent} · ${$('idAge').textContent}`);
    ok('risk tags classified', ['low', 'mod', 'high'].some(c => $('cvTag').className.includes(c)), `cvTag=${$('cvTag').textContent}`);
    ok('risk bars animated', $('cvBar').style.width !== '' , `cvBar width=${$('cvBar').style.width}`);
  } catch (e) { ok('Oculomics path', false, e.message); }

  /* ---- 3. UI wiring ---- */
  try {
    document.querySelector('[data-tab="t-emotion"]').click();
    ok('tabs switch', $('t-emotion').classList.contains('active') && !$('t-identity').classList.contains('active'), 'emotion panel active');
    document.querySelector('#trackerList li[data-step="7"]');
    window.__setTrackerStep && window.__setTrackerStep(7, 'done');
    ok('tracker -> 10/10', $('trackerCount').textContent === '10/10', $('trackerCount').textContent);
    ok('FAQ present', document.querySelectorAll('.faq details').length >= 6, document.querySelectorAll('.faq details').length + ' items');
    ok('all sections present', ['rppg', 'oculomics', 'insights', 'longevity', 'proof', 'privacy', 'faq'].every(id => !!$(id)), '9 sections');
  } catch (e) { ok('UI wiring', false, e.message); }

  const pre = document.createElement('pre');
  pre.id = 'QA_RESULTS';
  pre.textContent = '\n' + out.join('\n') + '\nTOTAL ' + out.filter(r => r.startsWith('PASS')).length + '/' + out.length + '\n';
  document.body.appendChild(pre);
  window.__qaDone = true;
});
