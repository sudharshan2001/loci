// Loci viewer: point cloud, camera path, highlighted answer, evidence panel.
// Loci stores positions in OpenCV axes (x right, y down, z forward); three.js uses y up, so (x, -y, -z).
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { PLYLoader } from "three/addons/loaders/PLYLoader.js";

const ACCENT = new THREE.Color("#ffc850");
const $ = (id) => document.getElementById(id);
const cv = (p) => new THREE.Vector3(p[0], -p[1], -p[2]);

// ---- three.js setup -------------------------------------------------------------------------------------
const canvas = $("view");
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
const scene3 = new THREE.Scene();
scene3.background = new THREE.Color("#0b0e13");
const camera = new THREE.PerspectiveCamera(55, 1, 0.01, 500);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;

let cloud = null, cloudBase = null, layers = new THREE.Group(), hit = new THREE.Group();
scene3.add(layers, hit);
let current = null;           // current scene metadata from /scene/{sid}
let tags = [];                // DOM labels anchored to 3D points
let fly = null;               // camera animation state

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}
addEventListener("resize", resize);

function addTag(pos, text, cls) {
  const el = document.createElement("div");
  el.className = `tag ${cls || ""}`;
  el.textContent = text;
  $("stage").appendChild(el);
  tags.push({ el, pos });
}
function clearTags(cls) {
  tags = tags.filter((t) => {
    if (!cls || t.el.classList.contains(cls)) { t.el.remove(); return false; }
    return true;
  });
}
function placeTags() {
  const r = canvas.getBoundingClientRect();
  for (const t of tags) {
    const p = t.pos.clone().project(camera);
    const vis = p.z < 1 && Math.abs(p.x) < 1.1 && Math.abs(p.y) < 1.1;
    t.el.style.display = vis ? "block" : "none";
    t.el.style.left = `${(p.x + 1) / 2 * r.width}px`;
    t.el.style.top = `${(1 - p.y) / 2 * r.height}px`;
  }
}

function animate(now) {
  requestAnimationFrame(animate);
  if (fly) {
    const k = Math.min(1, (now - fly.t0) / fly.ms), e = k * k * (3 - 2 * k);
    camera.position.lerpVectors(fly.fromPos, fly.toPos, e);
    controls.target.lerpVectors(fly.fromTgt, fly.toTgt, e);
    if (k >= 1) fly = null;
  }
  controls.update();
  renderer.render(scene3, camera);
  placeTags();
}

function flyTo(target, dist) {
  const dir = camera.position.clone().sub(controls.target).normalize();
  if (dir.lengthSq() < 1e-6) dir.set(0, 0.5, 1).normalize();
  dir.y = Math.max(dir.y, 0.45); dir.normalize();
  fly = { t0: performance.now(), ms: 1400, fromPos: camera.position.clone(), fromTgt: controls.target.clone(),
          toPos: target.clone().add(dir.multiplyScalar(dist)), toTgt: target.clone() };
}

// ---- sessions -------------------------------------------------------------------------------------------
async function api(path, opts) {
  const r = await fetch(path, opts);
  if (!r.ok) throw new Error(`${path}: ${r.status} ${await r.text()}`);
  return r.json();
}

async function loadSessions() {
  const list = await api("/sessions");
  const sel = $("session");
  sel.innerHTML = "";
  for (const s of list) {
    const o = document.createElement("option");
    o.value = s.session_id;
    o.textContent = `${s.scene === "street" ? "Street" : "Home"} · ${s.session_id} (${s.n_objects})`;
    sel.appendChild(o);
  }
  sel.onchange = () => loadSession(sel.value);
  if (list.length) await loadSession(list[list.length - 1].session_id);
}

async function loadSession(sid) {
  if (current && current.session_id === sid) return;
  $("session").value = sid;
  layers.clear(); hit.clear(); clearTags();
  if (cloud) { scene3.remove(cloud); cloud.geometry.dispose(); cloud = null; }
  current = await api(`/scene/${sid}`);

  const geo = await new PLYLoader().loadAsync(`/scene/${sid}.ply`);
  const pos = geo.attributes.position.array;
  for (let i = 0; i < pos.length; i += 3) { pos[i + 1] = -pos[i + 1]; pos[i + 2] = -pos[i + 2]; }
  geo.computeBoundingSphere();
  cloudBase = geo.attributes.color.array.slice();
  cloud = new THREE.Points(geo, new THREE.PointsMaterial({ size: current.scene === "street" ? 0.03 : 0.008,
                                                           vertexColors: true, sizeAttenuation: true }));
  scene3.add(cloud);

  // camera path and "you are here"
  const path = current.camera_path.map(cv);
  layers.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(path),
                            new THREE.LineBasicMaterial({ color: 0x5b6b80, transparent: true, opacity: 0.8 })));
  const here = cv(current.you_are_here.slice(0, 3).map((r) => r[3]));
  const me = new THREE.Mesh(new THREE.SphereGeometry(current.scene === "street" ? 0.25 : 0.025, 20, 20),
                            new THREE.MeshBasicMaterial({ color: 0x4fd18b }));
  me.position.copy(here);
  layers.add(me);
  addTag(here, "you are here", "");

  // home: a faint floor grid, so a robot's route has something to stand on
  if (current.floor_y != null) {
    const r = geo.boundingSphere.radius;
    const grid = new THREE.GridHelper(Math.ceil(r * 2.5), Math.ceil(r * 2.5) * 4, 0x2a3442, 0x1a2029);
    grid.position.set(geo.boundingSphere.center.x, -current.floor_y, geo.boundingSphere.center.z);
    layers.add(grid);
  }

  // street: a marker and name for every read shop sign
  if (current.scene === "street") {
    for (const o of current.objects.filter((o) => o.text)) {
      const p = cv(o.centroid);
      const pin = new THREE.Mesh(new THREE.SphereGeometry(0.25, 12, 12), new THREE.MeshBasicMaterial({ color: 0xe6ebf2 }));
      pin.position.copy(p);
      layers.add(pin);
      addTag(p, `${o.text} · ${Math.round(o.distance_along_walk_m)} m`, "shop");
    }
  }

  const c = geo.boundingSphere.center, r = geo.boundingSphere.radius;
  // frame the whole scene from behind and above where the walk started
  const start = cv(current.camera_path[0]);
  const back = start.clone().sub(c).setY(0).normalize();
  if (back.lengthSq() < 1e-6) back.set(0, 0, 1);
  camera.position.copy(c.clone().add(back.multiplyScalar(r * 1.1)).add(new THREE.Vector3(0, r * 0.7, 0)));
  controls.target.copy(c);
  camera.near = r / 500; camera.far = r * 20; camera.updateProjectionMatrix();
  $("legend").textContent = `${current.objects.length} objects · ${geo.attributes.position.count.toLocaleString()} points`;
  await loadPrivacy(sid);
  await loadAlerts(sid);
}

async function loadAlerts(sid) {
  const alerts = await api(`/alerts/${sid}`);
  $("alerts").hidden = !alerts.length;
  const fmt = (t) => (t == null ? "" : `${Math.floor(t / 60)}:${String(Math.round(t % 60)).padStart(2, "0")}`);
  $("alertlist").innerHTML = "";
  for (const a of alerts) {
    const li = document.createElement("li");
    li.innerHTML = `${a.text}<small>${a.kind} · seen at ${fmt(a.seen_at)}</small>`;
    // an alert is shown like an answer: highlight the object, show its evidence
    li.onclick = async () => {
      const ans = { question: "Leaving check", answer: a.text, object_id: a.object_id, evidence: a.evidence,
                    session: sid, seen_at: fmt(a.seen_at), verified: true, confidence: 1, used_cloud: false };
      if (current.scene === "home") {
        const nav = await api(`/navigate/${sid}/${a.object_id.split(":").pop()}`).catch(() => null);
        if (nav && nav.ok) ans.robot_goal = { waypoints_m: nav.waypoints_m, path_length_m: nav.length_m };
      }
      showAnswer(ans);
      highlight(ans);
    };
    $("alertlist").appendChild(li);
  }
}

async function loadPrivacy(sid) {
  const p = await api(`/privacy/${sid}`);
  const names = { "face": "faces", "license plate": "number plates", "computer screen": "screens",
                  "phone screen": "phone screens", "document with text": "documents" };
  $("ledger").innerHTML = Object.entries(p.blurred).map(([k, v]) =>
    `<div class="stat"><b>${v}</b><span>${names[k] || k} blurred</span></div>`).join("") +
    `<div class="stat"><b>${p.n_keyframes}</b><span>frames stored, all redacted</span></div>`;
}

// ---- answers --------------------------------------------------------------------------------------------
function highlight(ans) {
  hit.clear(); clearTags("hit");
  if (cloud) cloud.geometry.attributes.color.array.set(cloudBase);
  if (!ans.object_id || !current) return;
  const local = ans.object_id.split(":").pop();
  const o = current.objects.find((x) => x.object_id === local);
  if (!o) return;
  const a = cv(o.box_min), b = cv(o.box_max);
  const box = new THREE.Box3(a.clone().min(b), a.clone().max(b)).expandByScalar(0.02);

  // recolour the object's points
  const pos = cloud.geometry.attributes.position.array, col = cloud.geometry.attributes.color.array;
  const v = new THREE.Vector3();
  for (let i = 0; i < pos.length; i += 3) {
    v.set(pos[i], pos[i + 1], pos[i + 2]);
    if (box.containsPoint(v)) { col[i] = ACCENT.r; col[i + 1] = ACCENT.g; col[i + 2] = ACCENT.b; }
  }
  cloud.geometry.attributes.color.needsUpdate = true;
  hit.add(new THREE.Box3Helper(box, ACCENT));

  // path from "you are here" to the object (along the walk on the street)
  const target = cv(o.centroid);
  const here = cv(current.you_are_here.slice(0, 3).map((r) => r[3]));
  let pts = [here, target];
  if (current.scene === "street") {
    const cams = current.camera_path.map(cv);
    let k = 0, best = Infinity;
    cams.forEach((c, i) => { const d = c.distanceTo(target); if (d < best) { best = d; k = i; } });
    pts = [...cams.slice(k).reverse(), target];
  }
  const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
                              new THREE.LineDashedMaterial({ color: ACCENT, dashSize: 0.05, gapSize: 0.03 }));
  line.computeLineDistances();
  hit.add(line);
  addTag(target, o.text || o.label, "hit");
  const route = drawRobotPath(ans.robot_goal);
  drawPhoto(ans);
  // frame the object and, when there is one, the robot's route on the floor
  const frame = box.clone();
  if (route) route.forEach((p) => frame.expandByPoint(p));
  const size = frame.getSize(new THREE.Vector3()).length();
  flyTo(frame.getCenter(new THREE.Vector3()), Math.max(current.scene === "street" ? 6 : 1.2, size * (route ? 1.6 : 3)));
}

// a ground robot's route over the floor grid, slightly above the floor so it stays visible
function drawRobotPath(goal) {
  if (!goal || !goal.waypoints_m || goal.waypoints_m.length < 2) return null;
  const pts = goal.waypoints_m.map((w) => cv([w[0], w[1] - 0.02, w[2]]));
  hit.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts),
                         new THREE.LineBasicMaterial({ color: 0x4fd18b, linewidth: 2 })));
  const stand = new THREE.Mesh(new THREE.CylinderGeometry(0.12, 0.12, 0.02, 24),
                               new THREE.MeshBasicMaterial({ color: 0x4fd18b, transparent: true, opacity: 0.7 }));
  stand.position.copy(pts[pts.length - 1]);
  hit.add(stand);
  addTag(pts[pts.length - 1], `robot stops here · ${goal.path_length_m} m`, "hit");
  return pts;
}

// "see what you saw": the evidence photo, hung where the camera was when it was taken
let photoView = null;
function drawPhoto(ans) {
  photoView = null;
  const ev = (ans.evidence || [])[0];
  if (!ev || !current.cameras) return;
  const k = parseInt(ev.slice(4, 8), 10);
  const cam = current.cameras.find((c) => c.keyframe === k);
  const pos = current.camera_path[current.cameras.indexOf(cam)];
  if (!cam || !pos) return;
  // camera axes in three.js coordinates (flip y and z of the OpenCV frame)
  const R = cam.R, col = (j) => new THREE.Vector3(R[0][j], -R[1][j], -R[2][j]);
  const right = col(0), down = col(1), fwd = col(2);
  const origin = cv(pos);
  const d = current.scene === "street" ? 1.2 : 0.25;
  const hh = d * Math.tan((cam.fov_y_deg * Math.PI) / 360), hw = hh * cam.aspect;
  const centre = origin.clone().add(fwd.clone().multiplyScalar(d));
  const corner = (sx, sy) => centre.clone().add(right.clone().multiplyScalar(sx * hw)).add(down.clone().multiplyScalar(sy * hh));
  const c = [corner(-1, -1), corner(1, -1), corner(1, 1), corner(-1, 1)];
  const lines = [origin, c[0], origin, c[1], origin, c[2], origin, c[3], c[0], c[1], c[1], c[2], c[2], c[3], c[3], c[0]];
  hit.add(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(lines),
                                 new THREE.LineBasicMaterial({ color: 0xe6ebf2, transparent: true, opacity: 0.7 })));
  const tex = new THREE.TextureLoader().load(`/evidence/${ans.session}/${ev}.jpg?full=1`);
  tex.colorSpace = THREE.SRGBColorSpace;
  const plane = new THREE.Mesh(new THREE.PlaneGeometry(2 * hw, 2 * hh),
                               new THREE.MeshBasicMaterial({ map: tex, side: THREE.DoubleSide, transparent: true, opacity: 0.92 }));
  plane.position.copy(centre);
  plane.quaternion.setFromRotationMatrix(new THREE.Matrix4().makeBasis(right, down.clone().negate(), fwd.clone().negate()));
  hit.add(plane);
  photoView = { pos: origin.clone().sub(fwd.clone().multiplyScalar(d * 0.6)), target: centre };
}

function viewFromPhoto() {
  if (!photoView) return;
  fly = { t0: performance.now(), ms: 1600, fromPos: camera.position.clone(), fromTgt: controls.target.clone(),
          toPos: photoView.pos, toTgt: photoView.target };
}

function showAnswer(ans) {
  $("empty").hidden = true;
  $("result").hidden = false;
  $("question").textContent = ans.question || "";
  $("answer").textContent = ans.answer;
  const conf = Math.round((ans.confidence || 0) * 100);
  $("badges").innerHTML = [
    ans.used_cloud ? `<span class="badge cloud">Muse Spark (cloud)</span>` : `<span class="badge local">Glimmer · on device</span>`,
    ans.verified ? `<span class="badge ok">verified</span>` : `<span class="badge unsure">not sure</span>`,
    `<span class="badge">${conf}% confidence</span>`,
    ans.latency_s ? `<span class="badge">${ans.latency_s}s</span>` : "",
  ].join("");

  const ev = (ans.evidence || [])[0];
  $("evidence").hidden = !ev;
  if (ev && ans.session) {
    $("evimg").src = `/evidence/${ans.session}/${ev}.jpg`;
    $("evcap").textContent = `Redacted frame${ans.seen_at ? ` · seen at ${ans.seen_at}` : ""} · ${ev}`;
  }
  const facts = [["where", ans.location_text], ["seen at", ans.seen_at],
                 ["position", ans.position_m && ans.position_m.map((x) => x.toFixed(2)).join(", ") + " m"]];
  $("facts").innerHTML = facts.filter(([, v]) => v).map(([k, v]) => `<dt>${k}</dt><dd>${v}</dd>`).join("");
  $("goal").hidden = !ans.robot_goal;
  $("goaljson").textContent = JSON.stringify(ans.robot_goal, null, 2);
  $("trace").innerHTML = (ans.trace || []).map((t) => t.tool
    ? `<li><code>${t.tool}(${Object.values(t.args).join(", ")})</code></li>`
    : `<li>verifier: ${t.verifier.join("; ")}</li>`).join("");
}

async function submit(promise) {
  $("askform").classList.add("busy");
  $("answer").textContent = "Thinking…";
  try {
    const ans = await promise;
    if (ans.session && (!current || current.session_id !== ans.session)) await loadSession(ans.session);
    showAnswer(ans);
    highlight(ans);
  } catch (e) {
    showAnswer({ answer: `Something went wrong: ${e.message}`, confidence: 0 });
  } finally {
    $("askform").classList.remove("busy");
  }
}

$("askform").onsubmit = (e) => {
  e.preventDefault();
  const q = $("q").value.trim();
  if (!q) return;
  submit(api("/ask", { method: "POST", headers: { "Content-Type": "application/json" },
                       body: JSON.stringify({ question: q }) }));
};
$("seeit").onclick = viewFromPhoto;
$("copy").onclick = (e) => { e.preventDefault(); navigator.clipboard.writeText($("goaljson").textContent); };

// voice questions: record, then POST to /ask_audio
let rec = null;
$("mic").onclick = async () => {
  if (rec) { rec.stop(); return; }
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
  const chunks = [];
  rec = new MediaRecorder(stream);
  rec.ondataavailable = (e) => chunks.push(e.data);
  rec.onstop = () => {
    stream.getTracks().forEach((t) => t.stop());
    $("mic").classList.remove("rec");
    rec = null;
    const fd = new FormData();
    fd.append("audio", new Blob(chunks, { type: "audio/webm" }), "q.webm");
    submit(api("/ask_audio", { method: "POST", body: fd }).then((a) => { $("q").value = a.question; return a; }));
  };
  rec.start();
  $("mic").classList.add("rec");
};

resize();
requestAnimationFrame(animate);
// ?session=<id> opens a session, ?q=<question> asks on load, ?alert=<n> opens the n-th alert
// (handy for demos and screenshots)
const params = new URLSearchParams(location.search);
loadSessions()
  .then(() => params.get("session") && loadSession(params.get("session")))
  .then(() => {
    const q = params.get("q");
    if (q) { $("q").value = q; $("askform").requestSubmit(); }
    if (params.get("view") === "photo") setTimeout(viewFromPhoto, 1600);
    const n = params.get("alert");
    const li = n !== null ? $("alertlist").children[Number(n)] : null;
    if (li) li.click();
  })
  .catch((e) => { $("legend").textContent = e.message; });
