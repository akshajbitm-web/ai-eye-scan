/* ============================================================
   AI Eye Scan — shared MediaPipe Face Mesh loader
   One WASM instance, one serialised send() queue, many subscribers.
   Used by both the rPPG engine and the iris engine so the two
   modules never load or drive the model twice.
   ============================================================ */
(function () {
  'use strict';
  if (window.FaceMeshShared) return;

  const subs = new Set();
  let meshPromise = null;
  let chain = Promise.resolve();

  function build() {
    if (typeof window.FaceMesh !== 'function' || window.__faceMeshFailed) {
      return Promise.reject(new Error('MediaPipe FaceMesh is not available'));
    }
    const mesh = new window.FaceMesh({
      locateFile: f => `vendor/face_mesh/${f}`
    });
    mesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,          // required: adds iris points 468-477
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.5
    });
    mesh.onResults(res => {
      subs.forEach(fn => { try { fn(res); } catch (e) { /* subscriber error */ } });
    });
    return Promise.resolve(mesh);
  }

  window.FaceMeshShared = {
    get() {
      if (!meshPromise) {
        meshPromise = build().catch(err => { meshPromise = null; throw err; });
      }
      return meshPromise;
    },
    subscribe(fn) {
      subs.add(fn);
      return () => subs.delete(fn);
    },
    // MediaPipe rejects overlapping send() calls — queue them.
    send(mesh, image) {
      if (!mesh) return Promise.reject(new Error('no mesh'));
      chain = chain.catch(() => {}).then(() => mesh.send({ image }));
      return chain;
    }
  };
})();
