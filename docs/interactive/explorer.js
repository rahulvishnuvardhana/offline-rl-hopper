// Grid explorer: pick any two of the 70 grid runs (and one of 5 evaluation episodes each), watch both
// replay in 3D on one shared clock. Trajectories are the logged evaluation episodes, exported and
// checked against the evaluation logs (scripts/export_trajectories.py); nothing here simulates.
(function () {
  const root = document.getElementById("explorer");
  if (!root || !window.THREE) return;
  const T3 = window.THREE;
  const FPS = 125;
  const CAPS = [[0.05, 0.2], [0.05, 0.225], [0.04, 0.25], [0.06, 0.195]]; // radius, half-length (hopper.xml)
  const COLORS = [0xf2f0ea, 0xf2701c, 0x33373e, 0xf2701c];
  const ROWS = [
    ["bc", null, 0], ["bc", null, 0.05], ["bc", null, 0.2], ["bc", null, 0.5], ["bc", null, 1],
    ["iql", 0.7, 0], ["iql", 0.7, 0.05], ["iql", 0.7, 0.2], ["iql", 0.7, 0.5], ["iql", 0.7, 1],
    ["iql", 0.5, 0.05], ["iql", 0.5, 0.5], ["iql", 0.9, 0.05], ["iql", 0.9, 0.5],
  ];
  const runName = (algo, tau, p, s) => algo === "bc" ? `bc_p${p}_s${s}` : (tau === 0.7 ? `iql_p${p}_s${s}` : `iql_p${p}_s${s}_tau${tau}`);
  const pct = p => `${Math.round(p * 100)}%`;
  const rowLabel = (algo, tau, p) => algo === "bc" ? `BC \u00b7 ${pct(p)} expert` : `IQL \u03c4 ${tau} \u00b7 ${pct(p)} expert`;
  const tip = document.getElementById("tip");
  let INDEX = null, active = "A", rate = 0.5, t = 0, playing = false, last = null;
  const cache = {};
  const slots = { A: null, B: null };

  // ---------- 3D viewport ----------
  // Scene styled after the MuJoCo showcase render (report/demo/hopper_showcase.xml): side view, slightly
  // closer (distance 2.7, elevation -7 degrees, 40 degree field of view), sky gradient, 0.33 m checker
  // floor, a straight-down shadow light plus fill and headlight, and a floor reflection. Graded dark like
  // MuJoCo's stock scene: slate-to-black sky, black and light-grey checker, strong reflection.
  function canvasTex(w, h, paint) {
    const c = document.createElement("canvas"); c.width = w; c.height = h; paint(c.getContext("2d"), w, h);
    const t = new T3.CanvasTexture(c); t.encoding = T3.sRGBEncoding; return t;
  }
  function checkerTexture() {
    const t = canvasTex(128, 128, (g) => {
      for (let i = 0; i < 2; i++) for (let j = 0; j < 2; j++) { g.fillStyle = (i + j) % 2 ? "#0b0c0d" : "#b8bdb8"; g.fillRect(i * 64, j * 64, 64, 64); }
      g.strokeStyle = "#2a2d2b"; g.lineWidth = 2; g.strokeRect(1, 1, 126, 126);
    });
    t.wrapS = t.wrapT = T3.RepeatWrapping; t.repeat.set(600, 600); t.anisotropy = 16; return t;
  }
  function skyTexture() {
    return canvasTex(4, 256, (g, w, h) => {
      const gr = g.createLinearGradient(0, 0, 0, h);
      gr.addColorStop(0, "#66809b"); gr.addColorStop(0.5, "#3a4a5c"); gr.addColorStop(1, "#151b22");
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
    });
  }
  const DIST = 2.7, ELEV = 7 * Math.PI / 180, LOOK_Z = 0.8;
  function makeView(canvas, studio) {
    const renderer = new T3.WebGLRenderer({ canvas, antialias: true, alpha: !!studio });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    renderer.shadowMap.enabled = true; renderer.shadowMap.type = T3.PCFSoftShadowMap;
    renderer.outputEncoding = T3.sRGBEncoding;
    renderer.toneMapping = T3.NoToneMapping;
    const scene = new T3.Scene();
    if (!studio) { scene.background = skyTexture(); scene.fog = new T3.Fog(0x1a2129, 6, 24); }
    scene.add(new T3.HemisphereLight(0xb7c7da, 0x202422, 0.28));
    const sun = new T3.DirectionalLight(0xffffff, 0.95); // MuJoCo's light: straight down, casts the shadow
    sun.position.set(0, 6, 0); sun.castShadow = true; sun.shadow.mapSize.set(2048, 2048);
    sun.shadow.radius = 4; sun.shadow.bias = -0.0005;
    Object.assign(sun.shadow.camera, { left: -2, right: 2, top: 2, bottom: -2, near: 0.5, far: 10 });
    scene.add(sun); scene.add(sun.target);
    const fill = new T3.DirectionalLight(0x9fb3c9, 0.22); fill.position.set(0.4, 0.5, 0.6); scene.add(fill);
    const camera = new T3.PerspectiveCamera(40, 4 / 3, 0.05, 80);
    const head = new T3.DirectionalLight(0xffffff, 0.4); camera.add(head); head.position.set(0, 0, 1); scene.add(camera);
    const ground = new T3.Mesh(new T3.PlaneGeometry(400, 400),
      new T3.MeshStandardMaterial({ map: checkerTexture(), roughness: 0.35, metalness: 0, transparent: true, opacity: 0.72 }));
    ground.rotation.x = -Math.PI / 2; ground.position.x = 180; ground.receiveShadow = true; ground.renderOrder = 1; scene.add(ground);
    if (studio) { // product-render look: shadow only, plus a faint grid that scrolls as the robot travels
      ground.material = new T3.ShadowMaterial({ opacity: 0.28 });
      const gt = canvasTex(512, 512, (g, w, h) => { // grid lines every 1/3 m on a 6 m patch, fading out
        g.strokeStyle = "rgba(120,128,138,0.55)"; g.lineWidth = 2;
        for (let i = 0; i <= 18; i++) { const v = i * w / 18; g.beginPath(); g.moveTo(v, 0); g.lineTo(v, h); g.stroke(); g.beginPath(); g.moveTo(0, v); g.lineTo(w, v); g.stroke(); }
        g.globalCompositeOperation = "destination-in";
        const rg = g.createRadialGradient(w / 2, h / 2, w * 0.08, w / 2, h / 2, w / 2);
        rg.addColorStop(0, "rgba(0,0,0,1)"); rg.addColorStop(1, "rgba(0,0,0,0)"); g.fillStyle = rg; g.fillRect(0, 0, w, h);
      });
      const grid = new T3.Mesh(new T3.PlaneGeometry(6, 6), new T3.MeshBasicMaterial({ map: gt, transparent: true, depthWrite: false }));
      grid.rotation.x = -Math.PI / 2; grid.position.y = 0.002; scene.add(grid);
      ground.userData.grid = grid; scene.userData.grid = grid;
    }
    // Studio environment for reflections: a sky-to-floor gradient with two soft light strips, so the
    // metal parts reflect something and read as metal.
    const envTex = canvasTex(512, 256, (g, w, h) => {
      const gr = g.createLinearGradient(0, 0, 0, h);
      gr.addColorStop(0, "#8a9cb0"); gr.addColorStop(0.45, "#3a4654"); gr.addColorStop(0.55, "#1c1f22"); gr.addColorStop(1, "#08090a");
      g.fillStyle = gr; g.fillRect(0, 0, w, h);
      g.fillStyle = "rgba(255,255,255,0.95)"; g.fillRect(60, 40, 140, 22); g.fillRect(300, 70, 160, 16);
    });
    envTex.mapping = T3.EquirectangularReflectionMapping;
    const pmrem = new T3.PMREMGenerator(renderer);
    scene.environment = pmrem.fromEquirectangular(envTex).texture;

    const M = {
      paint: new T3.MeshPhysicalMaterial({ color: 0xe8500a, roughness: 0.38, metalness: 0.0, clearcoat: 0.7, clearcoatRoughness: 0.15, envMapIntensity: 0.45 }),
      steel: new T3.MeshPhysicalMaterial({ color: 0x9aa1a8, roughness: 0.28, metalness: 0.92, envMapIntensity: 1.2 }),
      gunmetal: new T3.MeshPhysicalMaterial({ color: 0x4b5058, roughness: 0.38, metalness: 0.85, envMapIntensity: 1.0 }),
      chrome: new T3.MeshPhysicalMaterial({ color: 0xd9dde2, roughness: 0.12, metalness: 1, envMapIntensity: 1.4 }),
      rubber: new T3.MeshStandardMaterial({ color: 0x17191c, roughness: 0.9, metalness: 0 }),
      glow: new T3.MeshStandardMaterial({ color: 0x0f2a36, emissive: 0x19c6ff, emissiveIntensity: 1.6, roughness: 0.3 }),
    };
    const ROT_Z = new T3.Euler(Math.PI / 2, 0, 0); // cylinder axis along the joint axis (scene z)

    // One link: a group at the capsule centre, local +y along the capsule axis, length 2h.
    function link(parent, i, op, shadow) {
      const g = new T3.Group(); parent.add(g);
      const add = (geo, mat, x, y, z, rot) => {
        const m = new T3.Mesh(geo, op(mat)); m.position.set(x || 0, y || 0, z || 0);
        if (rot) m.rotation.copy(rot); m.castShadow = shadow; g.add(m); return m;
      };
      const [r, h] = CAPS[i];
      if (i === 0) { // torso: steel core, orange armour shell, sensor head with a glowing visor strip
        add(new T3.CylinderGeometry(r * 0.9, r * 0.9, 2 * h, 24), M.steel);
        add(new T3.CylinderGeometry(r * 1.5, r * 1.35, h * 1.25, 28, 1, false, -2.2, 4.4), M.paint, 0, h * 0.18, 0);
        add(new T3.BoxGeometry(r * 0.7, h * 1.1, r * 2.2), M.gunmetal, -r * 1.05, h * 0.15, 0);
        add(new T3.CylinderGeometry(r * 1.25, r * 1.4, h * 0.42, 28), M.gunmetal, 0, h * 0.95, 0);
        add(new T3.BoxGeometry(r * 0.5, h * 0.12, r * 1.9), M.glow, r * 1.15, h * 1.0, 0);
        add(new T3.TorusGeometry(r * 1.42, r * 0.12, 8, 28), M.chrome, 0, h * 0.72, 0, new T3.Euler(Math.PI / 2, 0, 0));
      } else if (i === 1) { // thigh: steel strut between two orange armour plates
        add(new T3.CylinderGeometry(r * 0.55, r * 0.55, 2 * h, 18), M.steel);
        for (const s of [-1, 1]) add(new T3.BoxGeometry(r * 1.9, 2 * h * 0.82, r * 0.42), M.paint, 0, 0, s * r * 0.75);
        add(new T3.BoxGeometry(r * 0.5, 2 * h * 0.7, r * 1.2), M.gunmetal, r * 0.95, 0, 0);
      } else if (i === 2) { // shin: polished piston rod in a gunmetal sleeve
        add(new T3.CylinderGeometry(r * 0.55, r * 0.55, 2 * h, 20), M.chrome);
        add(new T3.CylinderGeometry(r * 1.05, r * 0.95, h * 1.05, 24), M.gunmetal, 0, h * 0.42, 0);
        add(new T3.CylinderGeometry(r * 1.12, r * 1.12, h * 0.1, 24), M.paint, 0, h * 0.92, 0);
        add(new T3.CylinderGeometry(r * 0.8, r * 0.8, h * 0.08, 20), M.steel, 0, -h * 0.85, 0);
      } else { // foot: orange shell on a black rubber sole
        add(new T3.CapsuleGeometry(r * 0.8, 2 * h * 0.95, 8, 24), M.paint);
        add(new T3.BoxGeometry(r * 1.7, 2 * h * 1.02, r * 1.5), M.rubber, r * 0.55, 0, 0);
        add(new T3.BoxGeometry(r * 0.35, 2 * h * 0.6, r * 1.6), M.gunmetal, -r * 0.75, 0, 0);
      }
      return g;
    }
    // A joint actuator: a steel drum along the joint axis with orange end caps and a chrome bolt ring.
    function actuator(parent, radius, op, shadow) {
      const g = new T3.Group(); parent.add(g);
      const mk = (geo, mat, z) => { const m = new T3.Mesh(geo, op(mat)); m.rotation.copy(ROT_Z); m.position.z = z || 0; m.castShadow = shadow; g.add(m); return m; };
      mk(new T3.CylinderGeometry(radius, radius, radius * 2.6, 28), M.gunmetal);
      for (const s of [-1, 1]) {
        mk(new T3.CylinderGeometry(radius * 0.82, radius * 0.82, radius * 0.35, 28), M.paint, s * radius * 1.45);
        mk(new T3.TorusGeometry(radius * 0.6, radius * 0.08, 8, 24), M.chrome, s * radius * 1.65).rotation.set(0, 0, 0);
      }
      return g;
    }
    const build = (parent, reflect) => {
      const op = m => { if (!reflect) return m; const c = m.clone(); c.transparent = true; c.opacity = 0.32; c.side = T3.DoubleSide; c.depthWrite = false; return c; };
      const links = [0, 1, 2, 3].map(i => link(parent, i, op, !reflect));
      const joints = [0.075, 0.066, 0.058].map(r => actuator(parent, r, op, !reflect));
      return { links, joints };
    };
    const body = build(scene, false);
    const mirror = new T3.Group(); mirror.scale.y = -1; mirror.visible = !studio; scene.add(mirror);
    const refl = build(mirror, true);

    // Dust puffs where the foot touches down.
    const dustMat = new T3.MeshBasicMaterial({ color: 0xc9cfc8, transparent: true, opacity: 0, depthWrite: false });
    const dust = Array.from({ length: 18 }, () => { const m = new T3.Mesh(new T3.SphereGeometry(1, 10, 8), dustMat.clone()); m.visible = false; scene.add(m); return { m, life: 0, vx: 0, vy: 0, vz: 0 }; });
    return { renderer, scene, camera, sun, body, refl, dust, camX: 0, footDown: true, lastT: null };
  }
  function place(rig, row) {
    const ends = [];
    for (let i = 0; i < 4; i++) {
      const x = row[3 * i], z = row[3 * i + 1], a = row[3 * i + 2], h = CAPS[i][1];
      rig.links[i].position.set(x, z, 0); rig.links[i].rotation.set(0, 0, -a);
      ends.push([[x + Math.sin(a) * h, z + Math.cos(a) * h], [x - Math.sin(a) * h, z - Math.cos(a) * h]]);
    }
    const at = [ends[1][0], ends[1][1], ends[2][1]]; // hip, knee, ankle
    rig.joints.forEach((j, k) => { j.position.set(at[k][0], at[k][1], 0); j.rotation.set(0, 0, -row[3 * (k + 1) + 2]); });
  }
  function puff(view, x) {
    let n = 0;
    for (const d of view.dust) {
      if (d.life > 0 || n >= 6) continue;
      d.life = 1; d.m.visible = true; d.m.position.set(x + (Math.random() - 0.5) * 0.25, 0.03, (Math.random() - 0.5) * 0.2);
      d.vx = (Math.random() - 0.5) * 0.6; d.vy = 0.25 + Math.random() * 0.3; d.vz = (Math.random() - 0.5) * 0.4; n++;
    }
  }
  function stepDust(view, dt) {
    for (const d of view.dust) {
      if (d.life <= 0) continue;
      d.life -= dt * 1.6;
      if (d.life <= 0) { d.m.visible = false; continue; }
      d.m.position.x += d.vx * dt; d.m.position.y += d.vy * dt; d.m.position.z += d.vz * dt; d.vy *= 0.92;
      const s = 0.03 + (1 - d.life) * 0.09; d.m.scale.set(s, s * 0.7, s); d.m.material.opacity = 0.45 * d.life;
    }
  }
  function pose(view, row, forward) {
    place(view.body, row); place(view.refl, row);
    const footZ = row[10], down = footZ < 0.075;
    if (forward && down && !view.footDown) puff(view, row[9]);
    view.footDown = down;
    const now = performance.now(), dt = view.lastT == null ? 0 : Math.min(0.05, (now - view.lastT) / 1000);
    view.lastT = now; stepDust(view, dt);
    view.camX += (row[0] - view.camX) * 0.35;
    view.camera.position.set(view.camX, LOOK_Z + DIST * Math.sin(ELEV), DIST * Math.cos(ELEV));
    view.camera.lookAt(view.camX, LOOK_Z, 0);
    view.sun.position.set(view.camX, 6, 0); view.sun.target.position.set(view.camX, 0, 0);
  }
  function resize(view, canvas) {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (canvas.width !== Math.floor(w * view.renderer.getPixelRatio())) { view.renderer.setSize(w, h, false); view.camera.aspect = w / h; view.camera.updateProjectionMatrix(); }
  }


  // Joint pointers (hero only): each joint is projected to the canvas every frame, and its label sits at
  // the end of a short leader line, so the words move with the robot.
  const JNAMES = ["hip", "knee", "ankle"], JOFF = [[86, -26], [80, -2], [74, 22]];
  const SVGNS = "http://www.w3.org/2000/svg";
  const tmpV = new T3.Vector3();
  function drawLabels(layer, view) {
    if (!layer.childElementCount) JNAMES.forEach(n => {
      const line = document.createElementNS(SVGNS, "line"), dot = document.createElementNS(SVGNS, "circle"), txt = document.createElementNS(SVGNS, "text");
      dot.setAttribute("r", 3.5); txt.textContent = n;
      [line, dot, txt].forEach(e => { e.dataset.j = n; layer.appendChild(e); });
    });
    const w = layer.clientWidth, h = layer.clientHeight;
    layer.setAttribute("viewBox", `0 0 ${w} ${h}`);
    view.body.joints.forEach((jn, k) => {
      tmpV.copy(jn.position).project(view.camera);
      const x = (tmpV.x + 1) / 2 * w, y = (1 - tmpV.y) / 2 * h;
      const [ox, oy] = JOFF[k], tx = Math.min(x + ox, w - 48), ty = y + oy;
      const [line, dot, txt] = layer.querySelectorAll(`[data-j="${JNAMES[k]}"]`);
      line.setAttribute("x1", x); line.setAttribute("y1", y); line.setAttribute("x2", tx - 4); line.setAttribute("y2", ty - 4);
      dot.setAttribute("cx", x); dot.setAttribute("cy", y);
      txt.setAttribute("x", tx); txt.setAttribute("y", ty);
    });
  }

  // Hero: the robot replays a recorded hop (BC trained on expert-only data, run 3, test attempt 1),
  // half speed, looping, with the joint pointers following.
  (function hero() {
    const canvas = document.getElementById("hero-canvas"), layer = document.getElementById("hero-labels");
    if (!canvas || !layer) return;
    let view;
    try { view = makeView(canvas, true); } catch (e) { return; }
    const still = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    fetch("traj/bc_p1_s2.json").then(r => r.json()).then(eps => {
      const rows = eps["10000"].slice(120, 520);
      let k = 0, prev = null;
      const frame = now => {
        if (prev != null && !still) k += (now - prev) / 1000 * FPS * 0.5;
        prev = now;
        if (k >= rows.length - 1) { k = 0; view.camX = rows[0][0]; }
        const i = Math.floor(k), u = k - i, a = rows[i], b = rows[Math.min(i + 1, rows.length - 1)];
        const row = a.map((v, c) => v + (b[c] - v) * u);
        resize(view, canvas); pose(view, row, true);
        view.camera.position.set(view.camX + 1.55, 1.0, 2.95); view.camera.lookAt(view.camX + 0.05, 0.68, 0);
        const grid = view.scene.userData.grid; if (grid) grid.position.x = Math.round(view.camX * 3) / 3; // snaps to the lines, so they scroll
        view.renderer.render(view.scene, view.camera); drawLabels(layer, view);
        if (!still) requestAnimationFrame(frame);
      };
      view.camX = rows[0][0];
      requestAnimationFrame(frame);
    }).catch(() => {});
  })();

  // ---------- slots ----------
  function slotEl(k, id) { return root.querySelector(`[data-slot="${k}"] [data-f="${id}"]`); }
  const views = {};
  ["A", "B"].forEach(k => { const c = root.querySelector(`[data-slot="${k}"] canvas`); try { views[k] = makeView(c); } catch (e) { c.replaceWith(Object.assign(document.createElement("p"), { textContent: "3D view unavailable in this browser (WebGL is off)." })); } });

  async function load(k, name, seed) {
    if (!cache[name]) cache[name] = fetch(`traj/${name}.json`).then(r => r.json());
    const eps = await cache[name];
    const info = INDEX.runs[name];
    const ep = info.episodes.find(e => e.seed === seed) || info.episodes[0];
    slots[k] = { name, info, seed: ep.seed, ep, rows: eps[String(ep.seed)] };
    renderSlotHead(k); markGrid(); t = 0; draw(true);
  }
  function renderSlotHead(k) {
    const s = slots[k]; if (!s) return;
    const i = s.info;
    slotEl(k, "title").textContent = i.algo === "bc" ? `BC \u00b7 ${pct(i.p)} expert \u00b7 run ${i.seed + 1}` : `IQL \u03c4 ${i.tau} \u00b7 ${pct(i.p)} expert \u00b7 run ${i.seed + 1}`;
    slotEl(k, "run").textContent = `score ${i.run_score.toFixed(1)} \u00b7 falls in ${(i.fall_rate * 100).toFixed(0)}% of attempts`;
    const eps = slotEl(k, "eps"); eps.innerHTML = "";
    i.episodes.forEach(e => {
      const b = document.createElement("button"); b.type = "button";
      b.className = "ep" + (e.seed === s.seed ? " on" : "") + (e.fell ? " fell" : "");
      b.innerHTML = `<span>${e.seed - 10000 + 1}</span><small>${e.score.toFixed(0)}${e.fell ? " \u2193" : ""}</small>`;
      b.title = `test attempt ${e.seed - 10000 + 1}: score ${e.score}, ${e.steps} steps, ${e.fell ? "fell" : "no fall"}`;
      b.addEventListener("click", () => load(k, s.name, e.seed));
      eps.appendChild(b);
    });
  }
  function length() { return Math.max(1, ...["A", "B"].map(k => slots[k] ? slots[k].rows.length : 1)); }
  function draw(force) {
    const n = length();
    const f = Math.min(Math.floor(t), n - 1);
    ["A", "B"].forEach(k => {
      const s = slots[k], v = views[k]; if (!s) return;
      const i = Math.min(f, s.rows.length - 1), row = s.rows[i];
      const j = Math.min(i + 1, s.rows.length - 1), u = Math.min(1, Math.max(0, t - i));
      const smooth = i === j ? row : row.map((v0, c) => v0 + (s.rows[j][c] - v0) * u);
      if (v) { resize(v, v.renderer.domElement); pose(v, smooth, playing); v.renderer.render(v.scene, v.camera); }
      slotEl(k, "step").textContent = i;
      slotEl(k, "ret").textContent = row[12].toFixed(0);
      const deg = row[2] * 180 / Math.PI, el = slotEl(k, "tilt");
      el.textContent = (deg > 0 ? "+" : "") + deg.toFixed(1) + "\u00b0";
      el.style.color = Math.abs(row[2]) > 0.2 ? "var(--no)" : Math.abs(row[2]) > 0.15 ? "var(--warn)" : "";
      slotEl(k, "dist").textContent = row[0].toFixed(1) + " m";
      const badge = slotEl(k, "badge"), ended = f >= s.rows.length - 1 && s.rows.length > 1;
      badge.hidden = !ended; badge.className = "badge" + (s.ep.fell ? " fall" : "");
      badge.textContent = s.ep.fell ? `fell at step ${s.ep.steps}` : `${s.ep.steps} steps, no fall`;
    });
    const sc = root.querySelector("#ex-scrub"); sc.max = n - 1; sc.value = f;
    root.querySelector("#ex-scrub-label").textContent = `step ${f} / ${n - 1}`;
  }
  function tick(now) {
    if (!playing) return;
    if (last != null) t += (now - last) / 1000 * FPS * rate;
    last = now;
    if (t >= length() - 1) { t = length() - 1; setPlay(false); root.querySelector("#ex-play").innerHTML = "&#8635; Replay"; }
    draw(); if (playing) requestAnimationFrame(tick);
  }
  function setPlay(on) {
    playing = on; last = null;
    root.querySelector("#ex-play").innerHTML = on ? "&#10074;&#10074; Pause" : "&#9654; Play both";
    if (on) { if (t >= length() - 1) t = 0; requestAnimationFrame(tick); }
  }

  // ---------- grid ----------
  function shade(v) { // one-hue sequential ramp (blue), light = low score, dark = high
    const stops = [[0, [205, 226, 251]], [40, [134, 182, 239]], [80, [57, 135, 229]], [106, [16, 66, 129]]];
    for (let i = 1; i < stops.length; i++) if (v <= stops[i][0]) {
      const [a, ca] = stops[i - 1], [b, cb] = stops[i], u = (v - a) / (b - a);
      return `rgb(${ca.map((c, j) => Math.round(c + (cb[j] - c) * u)).join(",")})`;
    }
    return "rgb(16,66,129)";
  }
  function buildGrid() {
    const g = root.querySelector("#ex-grid"); g.innerHTML = "";
    const head = document.createElement("div"); head.className = "gh"; head.textContent = "setting"; g.appendChild(head);
    for (let s = 0; s < 5; s++) { const h = document.createElement("div"); h.className = "gh"; h.textContent = `run ${s + 1}`; g.appendChild(h); }
    ROWS.forEach(([algo, tau, p]) => {
      const l = document.createElement("div"); l.className = "gl"; l.innerHTML = `<i class="swatch" style="background:var(--${algo === "bc" ? "bc" : "iql"})"></i>${rowLabel(algo, tau, p)}`; g.appendChild(l);
      for (let s = 0; s < 5; s++) {
        const name = runName(algo, tau, p, s), info = INDEX.runs[name];
        const b = document.createElement("button"); b.type = "button"; b.className = "gc"; b.dataset.run = name;
        b.style.background = shade(Math.max(0, info.run_score));
        b.style.color = info.run_score > 70 ? "#fff" : "#10203a";
        b.innerHTML = `<span>${info.run_score.toFixed(0)}</span>${info.fall_rate >= 0.5 ? "<small>falls</small>" : ""}`;
        b.setAttribute("aria-label", `${rowLabel(algo, tau, p)}, run ${s + 1}: score ${info.run_score}`);
        b.addEventListener("click", () => load(active, name, 10000));
        b.addEventListener("pointermove", ev => { tip.innerHTML = `<b>${rowLabel(algo, tau, p)}, run ${s + 1}</b><br>score ${info.run_score.toFixed(1)}<br>falls in ${(info.fall_rate * 100).toFixed(0)}% of attempts<br>click to load into slot ${active}`; tip.hidden = false; tip.style.left = Math.min(ev.clientX + 14, innerWidth - 270) + "px"; tip.style.top = ev.clientY + 14 + "px"; });
        b.addEventListener("pointerleave", () => { tip.hidden = true; });
        g.appendChild(b);
      }
    });
  }
  function markGrid() {
    root.querySelectorAll(".gc").forEach(b => { b.dataset.mark = (slots.A && slots.A.name === b.dataset.run ? "A" : "") + (slots.B && slots.B.name === b.dataset.run ? "B" : ""); });
  }
  function cellRuns(algo, tau, p) { return [0, 1, 2, 3, 4].map(s => runName(algo, tau, p, s)).sort((a, b) => INDEX.runs[a].run_score_exact - INDEX.runs[b].run_score_exact); }
  const PRESETS = {
    "median-005": () => [cellRuns("bc", null, 0.05)[2], cellRuns("iql", 0.7, 0.05)[2], "BC vs IQL at a 5% expert share (typical runs)"],
    "iql-best-worst": () => [cellRuns("iql", 0.7, 0.2)[4], cellRuns("iql", 0.7, 0.2)[0], "IQL at 20% expert: its best run vs its worst run"],
    "tau": () => [cellRuns("iql", 0.5, 0.05)[2], cellRuns("iql", 0.9, 0.05)[2], "IQL at 5% expert: \u03c4 0.5 vs \u03c4 0.9 (typical runs)"],
    "bc-data": () => [cellRuns("bc", null, 0)[2], cellRuns("bc", null, 1)[2], "BC trained on weak-only vs expert-only data"],
    "p1": () => [cellRuns("bc", null, 1)[2], cellRuns("iql", 0.7, 1)[2], "BC vs IQL with expert-only data"],
  };
  root.querySelectorAll("[data-preset]").forEach(b => b.addEventListener("click", async () => {
    const [a, c, label] = PRESETS[b.dataset.preset]();
    root.querySelectorAll("[data-preset]").forEach(x => x.classList.toggle("on", x === b));
    root.querySelector("#ex-preset-label").textContent = label;
    setPlay(false); await Promise.all([load("A", a, 10000), load("B", c, 10000)]); t = 0; draw(); setPlay(true);
  }));
  root.querySelectorAll("[data-target]").forEach(b => b.addEventListener("click", () => {
    active = b.dataset.target; root.querySelectorAll("[data-target]").forEach(x => x.classList.toggle("on", x === b));
  }));
  root.querySelector("#ex-play").addEventListener("click", () => setPlay(!playing));
  root.querySelectorAll("[data-exrate]").forEach(b => b.addEventListener("click", () => {
    rate = parseFloat(b.dataset.exrate); root.querySelectorAll("[data-exrate]").forEach(x => x.classList.toggle("on", x === b));
  }));
  root.querySelector("#ex-scrub").addEventListener("input", ev => { setPlay(false); t = Number(ev.target.value); draw(); });
  window.addEventListener("resize", () => draw());

  fetch("traj/index.json").then(r => r.json()).then(async j => {
    INDEX = j; buildGrid();
    const [a, c, label] = PRESETS["median-005"]();
    root.querySelector("#ex-preset-label").textContent = label;
    await Promise.all([load("A", a, 10000), load("B", c, 10000)]);
    t = 0; draw();
  }).catch(() => { root.querySelector("#ex-grid").textContent = "Could not load the run index."; });
})();
