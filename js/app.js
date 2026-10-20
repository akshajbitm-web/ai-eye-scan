/* ============================================================
   AI Eye Scan — app shell
   navigation, reveal animations, tabs, glow effects, tracker
   ============================================================ */
(function () {
  'use strict';
  const $ = id => document.getElementById(id);
  const $$ = s => Array.from(document.querySelectorAll(s));

  /* ---------- mobile nav ---------- */
  const burger = $('navBurger'), links = $('navLinks');
  if (burger && links) {
    burger.addEventListener('click', () => links.classList.toggle('open'));
    links.addEventListener('click', e => {
      if (e.target.tagName === 'A') links.classList.remove('open');
    });
  }

  /* ---------- reveal on scroll ---------- */
  const reveals = $$('.reveal');
  if ('IntersectionObserver' in window) {
    const io = new IntersectionObserver(entries => {
      entries.forEach(en => {
        if (en.isIntersecting) { en.target.classList.add('in'); io.unobserve(en.target); }
      });
    }, { threshold: .12, rootMargin: '0px 0px -40px 0px' });
    reveals.forEach((el, i) => { el.style.transitionDelay = (i % 5) * 60 + 'ms'; io.observe(el); });
  } else {
    reveals.forEach(el => el.classList.add('in'));
  }

  /* ---------- scroll spy ---------- */
  const navAnchors = $$('.nav-links a');
  const sections = navAnchors
    .map(a => document.querySelector(a.getAttribute('href')))
    .filter(Boolean);
  if (sections.length && 'IntersectionObserver' in window) {
    const spy = new IntersectionObserver(entries => {
      entries.forEach(en => {
        if (!en.isIntersecting) return;
        navAnchors.forEach(a => a.classList.toggle('active', a.getAttribute('href') === '#' + en.target.id));
      });
    }, { rootMargin: '-45% 0px -50% 0px' });
    sections.forEach(s => spy.observe(s));
  }

  /* ---------- tabs ---------- */
  $$('.tab').forEach(t => t.addEventListener('click', () => {
    $$('.tab').forEach(x => x.classList.remove('active'));
    $$('.tab-panel').forEach(p => p.classList.remove('active'));
    t.classList.add('active');
    const panel = $(t.dataset.tab);
    if (panel) panel.classList.add('active');
  }));

  /* ---------- card cursor glow ---------- */
  $$('.info-card').forEach(card => card.addEventListener('pointermove', e => {
    const r = card.getBoundingClientRect();
    card.style.setProperty('--mx', ((e.clientX - r.left) / r.width * 100).toFixed(1) + '%');
    card.style.setProperty('--my', ((e.clientY - r.top) / r.height * 100).toFixed(1) + '%');
  }));

  /* ---------- build tracker (right column) ---------- */
  const tracker = $('tracker');
  const tgl = $('trackerToggle');
  if (tracker && tgl) {
    tgl.addEventListener('click', () => {
      const collapsed = tracker.classList.toggle('collapsed');
      tgl.textContent = collapsed ? '+' : '–';
    });
  }
  window.__setTrackerStep = function (step, state) {
    const li = document.querySelector(`#trackerList li[data-step="${step}"]`);
    if (li) li.className = state || 'done';
    const all = $$('#trackerList li');
    const done = all.filter(x => x.classList.contains('done')).length;
    const bar = $('trackerBar'), cnt = $('trackerCount');
    if (bar) bar.style.width = Math.round(done / all.length * 100) + '%';
    if (cnt) cnt.textContent = `${done}/${all.length}`;
  };

  /* ---------- live updates from the engines ---------- */
  window.addEventListener('rppg:update', e => {
    const d = e.detail;
    if (!d || d.well === undefined) return;
    document.body.classList.toggle('scanning', !!d.good);
  });
  window.addEventListener('oculomics:done', e => {
    const d = e.detail;
    if (d && d.biomarkers) document.body.classList.add('ocu-done');
  });

  /* ---------- live to-do update (build status, right column) ---------- */
  if (window.__setTrackerStep) window.__setTrackerStep(6, 'done');

  /* ---------- keep sticky-nav anchors aligned ---------- */
  const style = document.createElement('style');
  style.textContent = 'section[id],div[id]{scroll-margin-top:78px}';
  document.head.appendChild(style);

  /* ---------- keyboard nicety: Esc stops the camera ---------- */
  document.addEventListener('keydown', e => {
    if (e.key === 'Escape' && window.RPPG && window.RPPG.running) window.RPPG.stop(true);
  });
})();
