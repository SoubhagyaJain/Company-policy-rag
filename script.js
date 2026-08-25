/* ═══════════════════════════════════════════════════════════
   ENTERPRISE POLICY RAG — LUXURY INTERACTIVE ENGINE
   Three.js 3D WebGL Mesh × Live Simulator × Command Palette × Study Mode
   ═══════════════════════════════════════════════════════════ */

document.addEventListener('DOMContentLoaded', () => {

  // ── Audio Feedback System (Web Audio API) ──────────────────
  let audioCtx = null;
  let isSoundEnabled = false;

  try {
    const savedAudio = localStorage.getItem('rag-playbook-audio');
    isSoundEnabled = savedAudio === 'true';
  } catch (e) { /* ignore */ }

  function updateAudioIcons() {
    document.querySelectorAll('.sound-on, .hud-sound-on').forEach(el => {
      el.style.display = isSoundEnabled ? 'inline-block' : 'none';
    });
    document.querySelectorAll('.sound-off, .hud-sound-off').forEach(el => {
      el.style.display = isSoundEnabled ? 'none' : 'inline-block';
    });
  }
  updateAudioIcons();

  function playUiSound(type = 'click') {
    if (!isSoundEnabled) return;
    try {
      if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      }
      if (audioCtx.state === 'suspended') {
        audioCtx.resume();
      }

      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      const now = audioCtx.currentTime;

      osc.connect(gain);
      gain.connect(audioCtx.destination);

      if (type === 'click') {
        osc.type = 'sine';
        osc.frequency.setValueAtTime(800, now);
        osc.frequency.exponentialRampToValueAtTime(400, now + 0.04);
        gain.gain.setValueAtTime(0.06, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.04);
        osc.start(now);
        osc.stop(now + 0.04);
      } else if (type === 'open') {
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(320, now);
        osc.frequency.exponentialRampToValueAtTime(640, now + 0.08);
        gain.gain.setValueAtTime(0.08, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.08);
        osc.start(now);
        osc.stop(now + 0.08);
      } else if (type === 'success') {
        osc.type = 'sine';
        osc.frequency.setValueAtTime(520, now);
        osc.frequency.setValueAtTime(780, now + 0.06);
        gain.gain.setValueAtTime(0.08, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.14);
        osc.start(now);
        osc.stop(now + 0.14);
      }
    } catch (e) { /* audio fallback */ }
  }

  function toggleAudio() {
    isSoundEnabled = !isSoundEnabled;
    try {
      localStorage.setItem('rag-playbook-audio', isSoundEnabled);
    } catch (e) { /* ignore */ }
    updateAudioIcons();
    showToast(isSoundEnabled ? '🔊 Sound effects enabled' : '🔇 Sound effects muted');
    if (isSoundEnabled) playUiSound('success');
  }

  const audioToggleBtn = document.getElementById('audioToggle');
  const hudAudioToggleBtn = document.getElementById('hudAudioToggle');
  if (audioToggleBtn) audioToggleBtn.addEventListener('click', toggleAudio);
  if (hudAudioToggleBtn) hudAudioToggleBtn.addEventListener('click', toggleAudio);

  // ── Toast Notification System ──────────────────────────────
  const toastContainer = document.getElementById('toastContainer');

  function showToast(message, duration = 2800) {
    if (!toastContainer) return;
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `<span>${message}</span>`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(10px)';
      toast.style.transition = 'all 0.3s ease';
      setTimeout(() => toast.remove(), 300);
    }, duration);
  }

  // ── Light / Dark Mode Toggle ──────────────────────────────
  const themeToggleBtn = document.getElementById('themeToggle');
  const hudThemeToggleBtn = document.getElementById('hudThemeToggle');
  const THEME_STORAGE_KEY = 'rag-playbook-theme';

  function getPreferredTheme() {
    try {
      const saved = localStorage.getItem(THEME_STORAGE_KEY);
      if (saved) return saved;
    } catch (e) { /* ignore */ }
    return 'dark'; // Default to ultra-luxurious dark mode
  }

  function applyTheme(theme) {
    document.documentElement.setAttribute('data-theme', theme);
    document.body.classList.toggle('dark-theme', theme === 'dark');
    if (themeToggleBtn) {
      themeToggleBtn.setAttribute('aria-label', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
      themeToggleBtn.setAttribute('title', `Switch to ${theme === 'dark' ? 'light' : 'dark'} mode`);
    }
    if (window.updateThreeJsTheme) {
      window.updateThreeJsTheme(theme);
    }
  }

  let currentTheme = getPreferredTheme();
  applyTheme(currentTheme);

  function toggleTheme() {
    currentTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(currentTheme);
    try {
      localStorage.setItem(THEME_STORAGE_KEY, currentTheme);
    } catch (e) { /* ignore */ }
    playUiSound('click');
    showToast(currentTheme === 'dark' ? '🌙 Obsidian Dark Theme' : '☀️ Gallery Editorial Light Theme');
  }

  if (themeToggleBtn) themeToggleBtn.addEventListener('click', toggleTheme);
  if (hudThemeToggleBtn) hudThemeToggleBtn.addEventListener('click', toggleTheme);

  // ── Three.js 3D Neural Vector Embedding Scene ──────────────
  (function initThreeJsHero() {
    const canvas = document.getElementById('heroCanvas3d');
    if (!canvas || typeof THREE === 'undefined') return;

    const container = document.getElementById('hero3dWrapper');
    let width = container ? container.clientWidth : 560;
    let height = container ? container.clientHeight : 560;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(50, width / height, 0.1, 1000);
    camera.position.z = 240;

    const renderer = new THREE.WebGLRenderer({ canvas, alpha: true, antialias: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    const group = new THREE.Group();
    scene.add(group);

    // Generate 160 interconnected neural vector nodes on a sphere
    const particleCount = 160;
    const geometry = new THREE.BufferGeometry();
    const positions = new Float32Array(particleCount * 3);
    const radius = 88;

    for (let i = 0; i < particleCount; i++) {
      const phi = Math.acos(-1 + (2 * i) / particleCount);
      const theta = Math.sqrt(particleCount * Math.PI) * phi;

      const x = radius * Math.cos(theta) * Math.sin(phi) + (Math.random() - 0.5) * 14;
      const y = radius * Math.sin(theta) * Math.sin(phi) + (Math.random() - 0.5) * 14;
      const z = radius * Math.cos(phi) + (Math.random() - 0.5) * 14;

      positions[i * 3] = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;
    }

    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

    // Particle Points Material
    const pointColor = currentTheme === 'dark' ? 0xE07A4B : 0xC4653A;
    const pointMaterial = new THREE.PointsMaterial({
      color: pointColor,
      size: 3.5,
      transparent: true,
      opacity: 0.85
    });
    const pointsMesh = new THREE.Points(geometry, pointMaterial);
    group.add(pointsMesh);

    // Line Connections between close nodes
    const lineIndices = [];
    const maxDistance = 38;

    for (let i = 0; i < particleCount; i++) {
      for (let j = i + 1; j < particleCount; j++) {
        const dx = positions[i * 3] - positions[j * 3];
        const dy = positions[i * 3 + 1] - positions[j * 3 + 1];
        const dz = positions[i * 3 + 2] - positions[j * 3 + 2];
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
        if (dist < maxDistance) {
          lineIndices.push(i, j);
        }
      }
    }

    const lineGeometry = new THREE.BufferGeometry();
    lineGeometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    lineGeometry.setIndex(lineIndices);

    const lineColor = currentTheme === 'dark' ? 0x4A90E2 : 0x2C6B92;
    const lineMaterial = new THREE.LineBasicMaterial({
      color: lineColor,
      transparent: true,
      opacity: currentTheme === 'dark' ? 0.3 : 0.2
    });
    const lineMesh = new THREE.LineSegments(lineGeometry, lineMaterial);
    group.add(lineMesh);

    // Core pulsing inner wireframe
    const coreGeo = new THREE.IcosahedronGeometry(42, 1);
    const coreMat = new THREE.MeshBasicMaterial({
      color: currentTheme === 'dark' ? 0xE07A4B : 0xC4653A,
      wireframe: true,
      transparent: true,
      opacity: 0.25
    });
    const coreMesh = new THREE.Mesh(coreGeo, coreMat);
    group.add(coreMesh);

    // Mouse Interaction
    let targetRotationX = 0;
    let targetRotationY = 0;
    let isDragging = false;
    let prevMouseX = 0;
    let prevMouseY = 0;

    window.addEventListener('mousemove', (e) => {
      const mouseX = (e.clientX / window.innerWidth) * 2 - 1;
      const mouseY = (e.clientY / window.innerHeight) * 2 - 1;
      targetRotationY = mouseX * 0.6;
      targetRotationX = mouseY * 0.4;
    });

    canvas.addEventListener('mousedown', (e) => {
      isDragging = true;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    });

    window.addEventListener('mouseup', () => { isDragging = false; });
    window.addEventListener('mousemove', (e) => {
      if (isDragging) {
        const deltaX = e.clientX - prevMouseX;
        const deltaY = e.clientY - prevMouseY;
        group.rotation.y += deltaX * 0.008;
        group.rotation.x += deltaY * 0.008;
        prevMouseX = e.clientX;
        prevMouseY = e.clientY;
      }
    });

    // Theme Update callback
    window.updateThreeJsTheme = (theme) => {
      const isDark = theme === 'dark';
      pointMaterial.color.setHex(isDark ? 0xE07A4B : 0xC4653A);
      lineMaterial.color.setHex(isDark ? 0x4A90E2 : 0x2C6B92);
      lineMaterial.opacity = isDark ? 0.35 : 0.2;
      coreMat.color.setHex(isDark ? 0xE07A4B : 0xC4653A);
      coreMat.opacity = isDark ? 0.3 : 0.2;
    };

    // Animation Loop
    let clock = new THREE.Clock();
    function animate() {
      requestAnimationFrame(animate);
      const elapsed = clock.getElapsedTime();

      group.rotation.y += 0.0025;
      group.rotation.x += (targetRotationX - group.rotation.x) * 0.05;
      group.rotation.y += (targetRotationY - group.rotation.y) * 0.05;

      coreMesh.rotation.y = -elapsed * 0.15;
      coreMesh.rotation.x = elapsed * 0.1;

      renderer.render(scene, camera);
    }
    animate();

    window.addEventListener('resize', () => {
      if (!container) return;
      const w = container.clientWidth;
      const h = container.clientHeight;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
  })();

  // ── Animated Number Counters on Scroll ─────────────────────
  const counterElements = document.querySelectorAll('.hero-stat-number[data-count]');
  let countersAnimated = false;

  function runCounters() {
    if (countersAnimated) return;
    countersAnimated = true;

    counterElements.forEach(el => {
      const target = parseFloat(el.getAttribute('data-count'));
      const prefix = el.getAttribute('data-prefix') || '';
      const suffix = el.getAttribute('data-suffix') || '';
      const decimals = parseInt(el.getAttribute('data-decimals') || '0', 10);
      const duration = 1600;
      const startTime = performance.now();

      function updateNumber(now) {
        const progress = Math.min((now - startTime) / duration, 1);
        const easeProgress = 1 - Math.pow(1 - progress, 3); // cubic ease out
        const currentVal = target * easeProgress;
        el.textContent = `${prefix}${currentVal.toFixed(decimals)}${suffix}`;

        if (progress < 1) {
          requestAnimationFrame(updateNumber);
        } else {
          el.textContent = `${prefix}${target.toFixed(decimals)}${suffix}`;
        }
      }
      requestAnimationFrame(updateNumber);
    });
  }

  const heroStatsGrid = document.querySelector('.hero-stats-grid');
  if (heroStatsGrid && 'IntersectionObserver' in window) {
    const statsObserver = new IntersectionObserver((entries) => {
      entries.forEach(entry => {
        if (entry.isIntersecting) {
          runCounters();
          statsObserver.unobserve(entry.target);
        }
      });
    }, { threshold: 0.2 });
    statsObserver.observe(heroStatsGrid);
  } else {
    runCounters();
  }

  // ── Interactive Live RAG Pipeline Simulator ────────────────
  (function initRAGSimulator() {
    const simQueryInput = document.getElementById('simQueryInput');
    const simRunBtn = document.getElementById('simRunBtn');
    const simStepNextBtn = document.getElementById('simStepNextBtn');
    const simResetBtn = document.getElementById('simResetBtn');
    const presetBtns = document.querySelectorAll('.sim-preset-btn');
    const speedBtns = document.querySelectorAll('.sim-speed-btn');
    const stepNodes = document.querySelectorAll('.sim-step-node');
    const traceLog = document.getElementById('simTraceLog');
    const verifierCard = document.getElementById('simVerifierCard');
    const outputBody = document.getElementById('simOutputBody');
    const citationsFooter = document.getElementById('simCitationsFooter');
    const citationsList = document.getElementById('simCitationsList');
    const totalLatencyEl = document.getElementById('simTotalLatency');
    const simIndicator = document.getElementById('simIndicator');

    // HUD Elements
    const hudTTFT = document.getElementById('hudTTFT');
    const hudTokensSec = document.getElementById('hudTokensSec');
    const hudFaithfulness = document.getElementById('hudFaithfulness');
    const hudModel = document.getElementById('hudModel');
    const hudGPU = document.getElementById('hudGPU');

    if (!simRunBtn || !simQueryInput) return;

    let simSpeed = '1'; // '1', '2', 'step'
    let activePreset = '1';
    let isSimulating = false;
    let stepResolve = null;

    const PRESETS = {
      '1': {
        query: "What is the paid parental leave policy for remote software engineers with over 2 years of company tenure?",
        intent: "FACTUAL_POLICY (Confidence: 0.982)",
        model: "qwen2.5:7b (GPU CUDA)",
        cache: "MISS (cos_sim: 0.814 < 0.950)",
        retrieved: [
          { code: "POL-HR-2024 §4.1", score: "0.912", text: "Tenured full-time employees (>24 months) receive 16 weeks of 100% paid parental leave, applicable to remote and international hubs." },
          { code: "POL-HR-2024 §4.3", score: "0.854", text: "Primary and secondary caregivers may split leave into two blocks within the first 12 months following birth or adoption." }
        ],
        rrf: "80 Candidates fused (Dense + BM25, k=60)",
        rerankTop: "Top 2 chunks passed relative threshold ratio (score: 0.962)",
        parentDoc: "Expanded to Parent Doc POL-HR-2024 §4 (2,048 tokens)",
        scores: { faith: 0.984, comp: 0.940, cit: 1.0, coh: 0.965, compScore: 0.968 },
        ttft: "178 ms",
        tps: "52.4 t/s",
        answer: "Under **POL-HR-2024 §4.1**, full-time software engineers with more than 2 years (>24 months) of company tenure are entitled to **16 weeks of 100% paid parental leave**. This policy applies uniformly to all distributed and remote engineering staff regardless of local jurisdiction. Furthermore, pursuant to **§4.3**, caregivers may choose to split their leave allocation into up to two distinct blocks within the first 12 months following birth or legal adoption.",
        citations: ["[Source 1: POL-HR-2024 §4.1 (Tenure & Parental Leave)]", "[Source 2: POL-HR-2024 §4.3 (Caregiver Leave Splitting)]"]
      },
      '2': {
        query: "What is the annual reimbursement limit for home office 4K monitors and ergonomic chairs?",
        intent: "FINANCIAL_THRESHOLD (Confidence: 0.991)",
        model: "qwen2.5:7b (GPU CUDA)",
        cache: "MISS (cos_sim: 0.789 < 0.950)",
        retrieved: [
          { code: "POL-FIN-308 §2.4", score: "0.935", text: "Remote staff are granted an initial $1,500 setup stipend, plus an annual recurring $500 refresh allowance for peripheral hardware and ergonomic seating." }
        ],
        rrf: "45 Candidates fused (Dense + BM25 exact match on '$1,500', k=60)",
        rerankTop: "Top 1 chunk passed relative threshold ratio (score: 0.981)",
        parentDoc: "Expanded to Parent Doc POL-FIN-308 §2 (1,840 tokens)",
        scores: { faith: 0.995, comp: 0.960, cit: 1.0, coh: 0.980, compScore: 0.985 },
        ttft: "162 ms",
        tps: "54.1 t/s",
        answer: "According to **POL-FIN-308 §2.4**, remote engineering employees are eligible for an initial **$1,500 one-time home office setup stipend** upon hire. Thereafter, employees receive an **annual recurring reimbursement allowance of $500** for hardware peripherals, 4K monitors, and ergonomic chairs, reimbursable through the automated Expensify portal with itemized receipts.",
        citations: ["[Source 1: POL-FIN-308 §2.4 (Home Office Hardware Caps)]"]
      },
      '3': {
        query: "What are the strict anti-retaliation protections for employees reporting compliance violations anonymously?",
        intent: "LEGAL_COMPLIANCE (Confidence: 0.978)",
        model: "qwen2.5:7b (GPU CUDA)",
        cache: "MISS (cos_sim: 0.835 < 0.950)",
        retrieved: [
          { code: "POL-ETH-101 §8.2", score: "0.948", text: "Zero tolerance for retaliatory action against any whistleblower. All reports via the Ethics Hotline are encrypted and routed directly to the Board Audit Committee." }
        ],
        rrf: "60 Candidates fused (Dense + BM25, k=60)",
        rerankTop: "Top 2 chunks passed relative threshold ratio (score: 0.954)",
        parentDoc: "Expanded to Parent Doc POL-ETH-101 §8 (2,200 tokens)",
        scores: { faith: 0.975, comp: 0.930, cit: 1.0, coh: 0.955, compScore: 0.962 },
        ttft: "185 ms",
        tps: "50.8 t/s",
        answer: "Under **POL-ETH-101 §8.2**, the organization enforces a strict zero-tolerance mandate against any retaliatory action taken against whistleblowers. Submissions through the third-party Ethics Hotline are end-to-end encrypted and routed directly to the independent Board Audit Committee, completely bypassing executive management to guarantee absolute whistleblower anonymity and legal protection.",
        citations: ["[Source 1: POL-ETH-101 §8.2 (Whistleblower Charter & Non-Retaliation)]"]
      },
      '4': {
        query: "When a regional European GDPR policy conflicts with the Global Retention standard, which takes legal precedence?",
        intent: "CONFLICT_RESOLUTION (Confidence: 0.965)",
        model: "qwen2.5:7b (GPU CUDA)",
        cache: "MISS (cos_sim: 0.772 < 0.950)",
        retrieved: [
          { code: "POL-SEC-001 §1.3", score: "0.942", text: "In all instances of conflict between Global baseline retention rules and local statutory regulations (e.g. EU GDPR / German BDSG), the stricter local statutory clause supercedes the Global standard." }
        ],
        rrf: "50 Candidates fused (Dense + BM25, k=60)",
        rerankTop: "Top 1 chunk passed relative threshold ratio (score: 0.974)",
        parentDoc: "Expanded to Parent Doc POL-SEC-001 §1 (2,400 tokens)",
        scores: { faith: 0.985, comp: 0.950, cit: 1.0, coh: 0.970, compScore: 0.977 },
        ttft: "172 ms",
        tps: "51.6 t/s",
        answer: "Pursuant to **POL-SEC-001 §1.3 (Jurisdictional Hierarchy & Conflict Precedence)**, regional statutory regulations—specifically European GDPR and localized statutory privacy acts—**strictly supersede and take legal precedence** over Global baseline retention standards whenever a conflict arises. The more restrictive privacy constraint must always be enforced for local customer and employee data.",
        citations: ["[Source 1: POL-SEC-001 §1.3 (Statutory Precedence & Retention)]"]
      },
      '5': {
        query: "How does the ingestion pipeline process technical architecture diagrams and code screenshots from PDF documents?",
        intent: "PROCEDURAL_MULTIMODAL (Confidence: 0.988)",
        model: "qwen2.5vl:7b (Vision VLM)",
        cache: "MISS (cos_sim: 0.801 < 0.950)",
        retrieved: [
          { code: "POL-TECH-502 §3.1", score: "0.965", text: "Visual pages containing diagrams or code screenshots are classified at ingestion and processed via qwen2.5vl:7b. Extractions are saved to storage/vision_cache/ indexed by SHA-256." }
        ],
        rrf: "65 Candidates fused (Dense + BM25 exact match on 'diagrams', k=60)",
        rerankTop: "Top 2 visual chunks passed relative threshold ratio (score: 0.982)",
        parentDoc: "Expanded to Multimodal Doc POL-TECH-502 §3 (2,450 tokens)",
        scores: { faith: 0.990, comp: 0.965, cit: 1.0, coh: 0.975, compScore: 0.983 },
        ttft: "192 ms",
        tps: "49.5 t/s",
        answer: "As outlined in **POL-TECH-502 §3.1**, PDF ingestion utilizes a dual-model multimodal pipeline. During document parsing, a heuristic classifier identifies visual elements (`CODE_SCREENSHOT`, `DIAGRAM_ARCHITECTURE`, `TABLE_DATA`). These pages are rendered at high resolution and routed to **qwen2.5vl:7b** with specialized prompts to preserve verbatim code indentation and component dataflows. The extracted representations are cached in `storage/vision_cache/` indexed by SHA-256 for instant sub-millisecond retrieval.",
        citations: ["[Source 1: POL-TECH-502 §3.1 (Multimodal Vision Ingestion Pipeline)]"]
      },
      '6': {
        query: "Hello! How are you doing today?",
        intent: "CONVERSATIONAL_BYPASS (Confidence: 0.999)",
        model: "qwen2.5:7b (Direct)",
        cache: "BYPASS (retrieval_required: false)",
        retrieved: [],
        rrf: "Bypassed (0 candidates)",
        rerankTop: "Bypassed (0ms GPU)",
        parentDoc: "Conversational Context (0 tokens docstore)",
        scores: { faith: 1.0, comp: 1.0, cit: 1.0, coh: 1.0, compScore: 1.0 },
        ttft: "42 ms",
        tps: "62.0 t/s",
        answer: "Hello! I am the **Enterprise Policy RAG Assistant**, ready to assist you with compliance guidelines, internal HR policies, technical documentation, and IT equipment procedures. How can I assist you with company policies today?",
        citations: []
      }
    };

    speedBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        speedBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        simSpeed = btn.dataset.speed;
        simStepNextBtn.style.display = simSpeed === 'step' ? 'inline-flex' : 'none';
        playUiSound('click');
      });
    });

    presetBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        presetBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        activePreset = btn.dataset.preset;
        if (PRESETS[activePreset]) {
          simQueryInput.value = PRESETS[activePreset].query;
        }
        playUiSound('click');
      });
    });

    if (simStepNextBtn) {
      simStepNextBtn.addEventListener('click', () => {
        if (stepResolve) {
          stepResolve();
          stepResolve = null;
          playUiSound('click');
        }
      });
    }

    async function waitStage(ms) {
      if (simSpeed === 'step') {
        simStepNextBtn.classList.add('pulse');
        await new Promise(resolve => { stepResolve = resolve; });
        simStepNextBtn.classList.remove('pulse');
        return;
      }
      const delay = simSpeed === '2' ? Math.max(40, ms * 0.45) : ms;
      await new Promise(r => setTimeout(r, delay));
    }

    function resetSimulation() {
      stepNodes.forEach(node => {
        node.classList.remove('active', 'completed');
        const statusEl = node.querySelector('.sim-step-status');
        if (statusEl) statusEl.textContent = 'Idle';
      });
      traceLog.innerHTML = '<div class="sim-log-item sim-log--info">Ready to execute. Select a preset or type any custom query to inspect real-time execution.</div>';
      verifierCard.style.display = 'none';
      citationsFooter.style.display = 'none';
      totalLatencyEl.textContent = 'Total: 0ms';
      if (simIndicator) simIndicator.classList.remove('live');

      hudTTFT.textContent = '-- ms';
      hudTokensSec.textContent = '-- t/s';
      hudFaithfulness.textContent = '-- %';

      outputBody.innerHTML = `
        <div class="sim-output-placeholder">
          <div class="sim-placeholder-icon">🤖</div>
          <p>Click <strong>"Run Pipeline Simulation"</strong> to observe tokens streaming in real-time with verified citation anchors and telemetry metadata.</p>
        </div>
      `;
      isSimulating = false;
      simRunBtn.disabled = false;
    }

    if (simResetBtn) simResetBtn.addEventListener('click', () => {
      resetSimulation();
      playUiSound('click');
    });

    function analyzeCustomQuery(userQ) {
      const q = userQ.toLowerCase();
      if (/^(hi|hello|hey|good morning|thanks|thank you)\b/i.test(q)) {
        return PRESETS['6'];
      }
      if (/diagram|architecture|vision|ocr|screenshot|pdf visual/i.test(q)) {
        return PRESETS['5'];
      }
      if (/reimburse|stipend|monitor|equipment|dollar|\$|expense|budget/i.test(q)) {
        return PRESETS['2'];
      }
      if (/whistleblower|retaliation|ethics|violation|legal/i.test(q)) {
        return PRESETS['3'];
      }
      if (/gdpr|conflict|precedence|statutory|jurisdiction/i.test(q)) {
        return PRESETS['4'];
      }
      return PRESETS['1'];
    }

    simRunBtn.addEventListener('click', async () => {
      if (isSimulating) return;
      isSimulating = true;
      simRunBtn.disabled = true;
      if (simIndicator) simIndicator.classList.add('live');
      playUiSound('open');

      const userText = simQueryInput.value.trim();
      let data = PRESETS[activePreset];

      if (userText && (!PRESETS[activePreset] || userText !== PRESETS[activePreset].query)) {
        data = { ...analyzeCustomQuery(userText), query: userText };
      }

      resetSimulation();
      isSimulating = true;
      simRunBtn.disabled = true;
      if (simIndicator) simIndicator.classList.add('live');

      // Update HUD
      hudModel.textContent = data.model || 'qwen2.5:7b';
      hudGPU.textContent = data.intent.includes('Vision') ? '4.8 GB (qwen2.5vl)' : '3.4 GB VRAM';

      traceLog.innerHTML = '';
      function addLog(msg, type = 'info') {
        const item = document.createElement('div');
        item.className = `sim-log-item sim-log--${type}`;
        item.innerHTML = msg;
        traceLog.appendChild(item);
        traceLog.scrollTop = traceLog.scrollHeight;
      }

      function setStepActive(stepNum, statusText) {
        stepNodes.forEach(n => {
          const num = parseInt(n.dataset.simStep, 10);
          if (num < stepNum) {
            n.classList.remove('active');
            n.classList.add('completed');
            n.querySelector('.sim-step-status').textContent = 'Passed';
          } else if (num === stepNum) {
            n.classList.add('active');
            n.querySelector('.sim-step-status').textContent = statusText || 'Running';
          } else {
            n.classList.remove('active', 'completed');
            n.querySelector('.sim-step-status').textContent = 'Idle';
          }
        });
      }

      const isConversational = data.intent.includes('CONVERSATIONAL_BYPASS');

      // ── Step 1: Intent Router ──────────────────────────────
      setStepActive(1, 'Routing');
      addLog(`[01 ROUTER] Classifying query intent via compiled regex heuristics...`, 'info');
      await waitStage(220);
      addLog(`[01 ROUTER] Classification: <strong>${data.intent}</strong> (Latency: 0.4ms)`, 'success');
      totalLatencyEl.textContent = 'Total: 1.2ms';

      if (isConversational) {
        // Conversational fast-path
        addLog(`[01 ROUTER] Conversational Greeting Detected → Activating Conversational Bypass (retrieval_required: false)`, 'warning');
        for (let i = 2; i <= 7; i++) {
          setStepActive(i, 'Skipped');
          await waitStage(60);
        }
        hudTTFT.textContent = '42 ms';
        hudTokensSec.textContent = '62.0 t/s';
        hudFaithfulness.textContent = '100.0%';
      } else {
        // ── Step 2: Semantic Cache ───────────────────────────
        setStepActive(2, 'Checking');
        addLog(`[02 CACHE] Probing ChromaDB Semantic Cache collection (cosine threshold ≥ 0.950)...`, 'info');
        await waitStage(240);
        addLog(`[02 CACHE] Cache ${data.cache} → Advancing to hybrid retrieval`, 'warning');
        totalLatencyEl.textContent = 'Total: 16.4ms';

        // ── Step 3: Hybrid Retrieval ─────────────────────────
        setStepActive(3, 'Retrieving');
        addLog(`[03 RETRIEVAL] Launching parallel Dense (bge-small-en-v1.5) + Sparse (rank-bm25)...`, 'info');
        await waitStage(300);
        addLog(`[03 RETRIEVAL] Fetched 40 Dense candidates + 40 BM25 sparse matches (Latency: 21.8ms)`, 'success');
        totalLatencyEl.textContent = 'Total: 38.2ms';

        // ── Step 4: RRF Fusion ───────────────────────────────
        setStepActive(4, 'Merging');
        addLog(`[04 RRF] Merging ranking scores via Reciprocal Rank Fusion (k=60)...`, 'info');
        await waitStage(220);
        addLog(`[04 RRF] ${data.rrf}`, 'success');
        totalLatencyEl.textContent = 'Total: 44.8ms';

        // ── Step 5: Cross-Encoder Reranker ───────────────────
        setStepActive(5, 'Reranking');
        addLog(`[05 RERANKER] Executing 24-layer cross-attention on CUDA GPU (<code>bge-reranker-large</code>)...`, 'info');
        await waitStage(420);
        addLog(`[05 RERANKER] ${data.rerankTop} (GPU Latency: 78.4ms)`, 'success');
        totalLatencyEl.textContent = 'Total: 123.2ms';

        // ── Step 6: Parent Expansion ─────────────────────────
        setStepActive(6, 'Expanding');
        addLog(`[06 DOCSTORE] Expanding top-scoring child chunks to full parent section boundaries...`, 'info');
        await waitStage(200);
        addLog(`[06 DOCSTORE] ${data.parentDoc}`, 'success');
        totalLatencyEl.textContent = 'Total: 135.6ms';

        // ── Step 7: 4D Verifier ──────────────────────────────
        setStepActive(7, 'Verifying');
        addLog(`[07 VERIFIER] Running 4D Grounding Evaluation: Faithfulness (35%), Completeness (30%), Citation Validity (20%), Coherence (15%)...`, 'info');
        verifierCard.style.display = 'block';

        document.getElementById('valFaith').textContent = data.scores.faith.toFixed(2);
        document.getElementById('barFaith').style.width = `${data.scores.faith * 100}%`;
        document.getElementById('valComp').textContent = data.scores.comp.toFixed(2);
        document.getElementById('barComp').style.width = `${data.scores.comp * 100}%`;
        document.getElementById('valCit').textContent = data.scores.cit.toFixed(2);
        document.getElementById('barCit').style.width = `${data.scores.cit * 100}%`;
        document.getElementById('valCoh').textContent = data.scores.coh.toFixed(2);
        document.getElementById('barCoh').style.width = `${data.scores.coh * 100}%`;
        document.getElementById('simVerdict').textContent = `Composite Score: ${data.scores.compScore.toFixed(3)} (PASS ≥ 0.70)`;

        hudFaithfulness.textContent = `${(data.scores.faith * 100).toFixed(1)}%`;
        hudTTFT.textContent = data.ttft || '178 ms';
        hudTokensSec.textContent = data.tps || '52.4 t/s';

        await waitStage(320);
        addLog(`[07 VERIFIER] Quality Gate Passed: <strong>${data.scores.compScore.toFixed(3)} ≥ 0.70</strong> (Numerical grounding validated in 1.8ms)`, 'success');
        totalLatencyEl.textContent = 'Total: 142.5ms';
      }

      // ── Step 8: Token Streaming Output ─────────────────────
      setStepActive(8, 'Streaming');
      outputBody.innerHTML = '';
      const answerTokens = data.answer.split(' ');
      let currentOutput = '';

      const tokenInterval = simSpeed === '2' ? 8 : 20;

      for (let i = 0; i < answerTokens.length; i++) {
        const token = answerTokens[i];
        currentOutput += token + ' ';
        outputBody.innerHTML = `<div class="sim-stream-text">${currentOutput.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')} <span class="typing-cursor">▌</span></div>`;
        outputBody.scrollTop = outputBody.scrollHeight;

        // Natural human-like pause at punctuation marks
        let waitMs = tokenInterval;
        if (token.endsWith('.') || token.endsWith(';') || token.endsWith(':')) {
          waitMs += 35;
        } else if (token.endsWith(',')) {
          waitMs += 18;
        }
        await new Promise(r => setTimeout(r, waitMs));
      }

      outputBody.innerHTML = `<div class="sim-stream-text">${currentOutput.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}</div>`;

      // Citations
      if (data.citations && data.citations.length > 0) {
        citationsFooter.style.display = 'block';
        citationsList.innerHTML = data.citations.map(c => `<span class="sim-citation-pill">${c}</span>`).join('');
      }

      stepNodes.forEach(n => {
        n.classList.remove('active');
        n.classList.add('completed');
        n.querySelector('.sim-step-status').textContent = 'Done';
      });

      addLog(`[08 STREAM] Verified token stream complete. Citations validated. Dispatching telemetry to write-behind SQLite buffer.`, 'success');
      totalLatencyEl.textContent = isConversational ? 'Total: 42ms' : 'Total: 485ms';
      if (simIndicator) simIndicator.classList.remove('live');
      isSimulating = false;
      simRunBtn.disabled = false;
      playUiSound('success');
    });
  })();

  // ── Global Command Palette (⌘K) ────────────────────────────
  (function initCommandPalette() {
    const cmdkModal = document.getElementById('cmdkModal');
    const cmdkInput = document.getElementById('cmdkInput');
    const cmdkResults = document.getElementById('cmdkResults');
    const cmdkTabs = document.querySelectorAll('.cmdk-tab');
    const cmdkCount = document.getElementById('cmdkCount');
    const cmdkTriggers = document.querySelectorAll('.cmdk-trigger');

    if (!cmdkModal || !cmdkInput || !cmdkResults) return;

    let indexEntries = [];
    let activeFilter = 'all';
    let selectedIndex = 0;

    // Index all questions
    document.querySelectorAll('.question-block').forEach((block, idx) => {
      const qNum = block.querySelector('.question-num')?.textContent.trim() || `Q${idx + 1}`;
      const qText = block.querySelector('.question-text')?.textContent.trim() || '';
      const mod = block.dataset.module || 'mod1';
      indexEntries.push({
        type: 'qa',
        badge: qNum,
        title: qText,
        meta: `Module: ${mod.toUpperCase()}`,
        element: block
      });
    });

    // Index architecture nodes
    document.querySelectorAll('.arch-node').forEach((node) => {
      const title = node.querySelector('.arch-node-title')?.textContent.trim() || '';
      const desc = node.querySelector('.arch-node-desc')?.textContent.trim() || '';
      if (title) {
        indexEntries.push({
          type: 'arch',
          badge: 'ARCH',
          title: title,
          meta: desc,
          element: node
        });
      }
    });

    // Index Failure Scenarios
    document.querySelectorAll('.failure-card').forEach((card) => {
      const title = card.querySelector('.failure-title')?.textContent.trim() || '';
      if (title) {
        indexEntries.push({
          type: 'failures',
          badge: 'FAIL',
          title: title,
          meta: 'Failure Mode & Remediation',
          element: card
        });
      }
    });

    // Index Design Decisions
    document.querySelectorAll('.decision-card').forEach((card) => {
      const title = card.querySelector('.decision-title, h3')?.textContent.trim() || '';
      if (title) {
        indexEntries.push({
          type: 'decisions',
          badge: 'TRADE-OFF',
          title: title,
          meta: 'System Design Decision',
          element: card
        });
      }
    });

    function openCmdk() {
      cmdkModal.style.display = 'flex';
      cmdkInput.value = '';
      selectedIndex = 0;
      renderResults();
      setTimeout(() => cmdkInput.focus(), 50);
      playUiSound('open');
    }

    function closeCmdk() {
      cmdkModal.style.display = 'none';
      cmdkInput.blur();
    }

    cmdkTriggers.forEach(btn => btn.addEventListener('click', openCmdk));

    cmdkModal.addEventListener('click', (e) => {
      if (e.target === cmdkModal) closeCmdk();
    });

    function renderResults() {
      const query = cmdkInput.value.toLowerCase().trim();
      const filtered = indexEntries.filter(item => {
        const typeMatch = activeFilter === 'all' || item.type === activeFilter;
        const textMatch = !query || item.title.toLowerCase().includes(query) || item.meta.toLowerCase().includes(query) || item.badge.toLowerCase().includes(query);
        return typeMatch && textMatch;
      });

      if (cmdkCount) {
        cmdkCount.textContent = `${filtered.length} entries`;
      }

      if (filtered.length === 0) {
        cmdkResults.innerHTML = '<div class="cmdk-empty">No results found matching your query.</div>';
        return;
      }

      cmdkResults.innerHTML = filtered.slice(0, 30).map((item, idx) => `
        <div class="cmdk-item ${idx === selectedIndex ? 'selected' : ''}" data-index="${idx}">
          <div class="cmdk-item-left">
            <span class="cmdk-item-badge">${item.badge}</span>
            <span class="cmdk-item-text">${item.title}</span>
          </div>
          <span class="cmdk-item-meta">${item.meta}</span>
        </div>
      `).join('');

      // Add click listeners to items
      cmdkResults.querySelectorAll('.cmdk-item').forEach((itemEl, idx) => {
        itemEl.addEventListener('click', () => {
          selectItem(filtered[idx]);
        });
      });
    }

    function selectItem(item) {
      if (!item || !item.element) return;
      closeCmdk();

      // If it's a question block, expand it
      if (item.element.classList.contains('question-block')) {
        item.element.classList.remove('hidden');
        item.element.classList.add('open');
      }

      item.element.scrollIntoView({ behavior: 'smooth', block: 'center' });
      item.element.style.outline = '2px solid var(--accent-core)';
      item.element.style.boxShadow = '0 0 25px var(--accent-core)';
      setTimeout(() => {
        item.element.style.outline = '';
        item.element.style.boxShadow = '';
      }, 2000);
      playUiSound('success');
    }

    cmdkInput.addEventListener('input', () => {
      selectedIndex = 0;
      renderResults();
    });

    cmdkTabs.forEach(tab => {
      tab.addEventListener('click', () => {
        cmdkTabs.forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        activeFilter = tab.dataset.tab;
        selectedIndex = 0;
        renderResults();
        playUiSound('click');
      });
    });

    document.addEventListener('keydown', (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') {
        e.preventDefault();
        if (cmdkModal.style.display === 'flex') {
          closeCmdk();
        } else {
          openCmdk();
        }
      } else if (e.key === 'Escape' && cmdkModal.style.display === 'flex') {
        closeCmdk();
      } else if (cmdkModal.style.display === 'flex') {
        const items = cmdkResults.querySelectorAll('.cmdk-item');
        if (e.key === 'ArrowDown') {
          e.preventDefault();
          selectedIndex = (selectedIndex + 1) % Math.max(items.length, 1);
          renderResults();
        } else if (e.key === 'ArrowUp') {
          e.preventDefault();
          selectedIndex = (selectedIndex - 1 + items.length) % Math.max(items.length, 1);
          renderResults();
        } else if (e.key === 'Enter') {
          e.preventDefault();
          const query = cmdkInput.value.toLowerCase().trim();
          const filtered = indexEntries.filter(item => {
            const typeMatch = activeFilter === 'all' || item.type === activeFilter;
            const textMatch = !query || item.title.toLowerCase().includes(query) || item.meta.toLowerCase().includes(query);
            return typeMatch && textMatch;
          });
          if (filtered[selectedIndex]) {
            selectItem(filtered[selectedIndex]);
          }
        }
      }
    });
  })();

  // ── Question Bookmarks (Star ⭐) & Copying System ───────────
  (function initQuestionEnhancements() {
    const STARRED_STORAGE_KEY = 'rag-playbook-starred';
    let starredQuestions = {};

    try {
      const saved = localStorage.getItem(STARRED_STORAGE_KEY);
      if (saved) starredQuestions = JSON.parse(saved);
    } catch (e) { /* ignore */ }

    function updateStarredCount() {
      const count = Object.keys(starredQuestions).length;
      const countEl = document.getElementById('starredCount');
      if (countEl) countEl.textContent = count;
    }

    const questionBlocks = document.querySelectorAll('.question-block');

    questionBlocks.forEach((block, idx) => {
      const header = block.querySelector('.question-header');
      if (!header) return;

      const qNumEl = header.querySelector('.question-num');
      const qNum = qNumEl ? qNumEl.textContent.trim() : `Q${idx + 1}`;
      const isStarred = !!starredQuestions[qNum];

      // Create Actions Container in header
      const actionsDiv = document.createElement('div');
      actionsDiv.className = 'question-header-actions';

      // Star Button
      const starBtn = document.createElement('button');
      starBtn.className = `question-star-btn ${isStarred ? 'starred' : ''}`;
      starBtn.title = isStarred ? 'Remove from Starred' : 'Star for Revision';
      starBtn.innerHTML = isStarred ? '★' : '☆';

      starBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        if (starredQuestions[qNum]) {
          delete starredQuestions[qNum];
          starBtn.classList.remove('starred');
          starBtn.innerHTML = '☆';
          starBtn.title = 'Star for Revision';
          showToast(`Unstarred ${qNum}`);
        } else {
          starredQuestions[qNum] = true;
          starBtn.classList.add('starred');
          starBtn.innerHTML = '★';
          starBtn.title = 'Remove from Starred';
          showToast(`⭐ Starred ${qNum} for Quick Revision`);
          playUiSound('success');
        }

        try {
          localStorage.setItem(STARRED_STORAGE_KEY, JSON.stringify(starredQuestions));
        } catch (err) { /* ignore */ }
        updateStarredCount();
      });

      // Copy Button
      const copyBtn = document.createElement('button');
      copyBtn.className = 'question-copy-btn';
      copyBtn.title = 'Copy question and answer to clipboard';
      copyBtn.innerHTML = '📋';

      copyBtn.addEventListener('click', async (e) => {
        e.stopPropagation();
        const qText = block.querySelector('.question-text')?.textContent.trim() || '';
        const shortAns = block.querySelector('.qa-short-answer')?.textContent.trim() || '';
        const deepAns = block.querySelector('.qa-deep-answer')?.textContent.trim() || '';
        const codeSnippet = block.querySelector('.qa-code-block')?.textContent.trim() || '';

        let markdown = `### ${qNum}: ${qText}\n\n**Short Answer:**\n${shortAns}\n\n**Deep Technical Answer:**\n${deepAns}`;
        if (codeSnippet) {
          markdown += `\n\n\`\`\`python\n${codeSnippet}\n\`\`\``;
        }

        try {
          await navigator.clipboard.writeText(markdown);
          showToast(`📋 Copied ${qNum} to clipboard!`);
          playUiSound('click');
        } catch (err) {
          showToast('Failed to copy to clipboard');
        }
      });

      actionsDiv.appendChild(starBtn);
      actionsDiv.appendChild(copyBtn);

      // Insert before toggle button
      const toggleBtn = header.querySelector('.question-toggle');
      if (toggleBtn) {
        header.insertBefore(actionsDiv, toggleBtn);
      } else {
        header.appendChild(actionsDiv);
      }
    });

    updateStarredCount();

    // ── 100 Q&A Filtering & Search ───────────────────────────
    const qaSearchInput = document.querySelector('.qa-search-input');
    const qaClearBtn = document.getElementById('qaClearSearch');
    const qaPills = document.querySelectorAll('.qa-pill');
    const qaToggleAllBtn = document.getElementById('qaToggleAll');
    const qaPrintBtn = document.getElementById('qaPrintBtn');
    const qaVisibleCounter = document.querySelector('.qa-visible-count');
    const qaModuleHeaders = document.querySelectorAll('.qa-module-header');

    let activeModule = 'all';
    let searchTerm = '';

    function filterQuestions() {
      let visibleCount = 0;
      const term = searchTerm.toLowerCase().trim();

      questionBlocks.forEach(block => {
        const qNum = block.querySelector('.question-num')?.textContent.trim() || '';
        const isStarred = !!starredQuestions[qNum];
        let moduleMatch = false;

        if (activeModule === 'all') {
          moduleMatch = true;
        } else if (activeModule === 'starred') {
          moduleMatch = isStarred;
        } else {
          moduleMatch = (block.dataset.module === activeModule);
        }

        const textContent = (block.textContent || '').toLowerCase();
        const searchMatch = !term || textContent.includes(term);

        if (moduleMatch && searchMatch) {
          block.classList.remove('hidden');
          visibleCount++;
        } else {
          block.classList.add('hidden');
        }
      });

      qaModuleHeaders.forEach(header => {
        const modId = header.dataset.module;
        if (activeModule === 'all') {
          const hasVisible = document.querySelector(`.question-block[data-module="${modId}"]:not(.hidden)`);
          header.style.display = hasVisible ? 'flex' : 'none';
        } else if (activeModule === 'starred') {
          const hasVisible = document.querySelector(`.question-block[data-module="${modId}"]:not(.hidden)`);
          header.style.display = hasVisible ? 'flex' : 'none';
        } else {
          header.style.display = (modId === activeModule) ? 'flex' : 'none';
        }
      });

      if (qaVisibleCounter) {
        qaVisibleCounter.textContent = `Showing ${visibleCount} of ${questionBlocks.length} questions`;
      }

      if (qaClearBtn) {
        qaClearBtn.style.display = searchTerm ? 'block' : 'none';
      }
    }

    if (qaSearchInput) {
      qaSearchInput.addEventListener('input', (e) => {
        searchTerm = e.target.value;
        filterQuestions();
      });
    }

    if (qaClearBtn) {
      qaClearBtn.addEventListener('click', () => {
        qaSearchInput.value = '';
        searchTerm = '';
        filterQuestions();
        qaSearchInput.focus();
      });
    }

    qaPills.forEach(pill => {
      pill.addEventListener('click', () => {
        qaPills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        activeModule = pill.dataset.filter || 'all';
        filterQuestions();
        playUiSound('click');
      });
    });

    if (qaToggleAllBtn) {
      let allOpen = false;
      qaToggleAllBtn.addEventListener('click', () => {
        allOpen = !allOpen;
        const visibleBlocks = document.querySelectorAll('.question-block:not(.hidden)');
        visibleBlocks.forEach(b => b.classList.toggle('open', allOpen));
        qaToggleAllBtn.textContent = allOpen ? 'Collapse All' : 'Expand All';
        playUiSound('open');
      });
    }

    if (qaPrintBtn) {
      qaPrintBtn.addEventListener('click', () => {
        const visibleBlocks = document.querySelectorAll('.question-block:not(.hidden)');
        visibleBlocks.forEach(b => b.classList.add('open'));
        window.print();
      });
    }
  })();

  // ── Circular Scroll Progress & Floating HUD ────────────────
  const progressBar = document.querySelector('.nav-progress');
  const hudProgressBar = document.getElementById('hudProgressBar');
  const backToTopBtn = document.getElementById('backToTopBtn');

  window.addEventListener('scroll', () => {
    const scrollTop = window.scrollY;
    const docHeight = document.documentElement.scrollHeight - window.innerHeight;
    const progress = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;

    if (progressBar) progressBar.style.width = `${progress}%`;

    if (hudProgressBar) {
      const circumference = 94.2; // 2 * pi * 15
      const offset = circumference - (progress / 100) * circumference;
      hudProgressBar.style.strokeDashoffset = offset;
    }
  }, { passive: true });

  if (backToTopBtn) {
    backToTopBtn.addEventListener('click', () => {
      window.scrollTo({ top: 0, behavior: 'smooth' });
      playUiSound('click');
    });
  }

  // ── Depth Layers Accordion ─────────────────────────────────
  const depthLayers = document.querySelectorAll('.depth-layer');
  depthLayers.forEach(layer => {
    layer.addEventListener('click', () => {
      const wasActive = layer.classList.contains('active');
      depthLayers.forEach(l => l.classList.remove('active'));
      if (!wasActive) {
        layer.classList.add('active');
        playUiSound('click');
      }
    });
  });

  // ── Question Blocks Accordion ──────────────────────────────
  const questionBlocks = document.querySelectorAll('.question-block');
  questionBlocks.forEach(block => {
    const header = block.querySelector('.question-header');
    if (header) {
      header.addEventListener('click', (e) => {
        if (e.target.closest('.question-star-btn') || e.target.closest('.question-copy-btn')) return;
        block.classList.toggle('open');
        playUiSound(block.classList.contains('open') ? 'open' : 'click');
      });
    }
  });

  // ── Follow-up Accordion ────────────────────────────────────
  const followupItems = document.querySelectorAll('.followup-item');
  followupItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.stopPropagation();
      item.classList.toggle('open');
      playUiSound('click');
    });
  });

  // ── Execution Timeline Steps ───────────────────────────────
  const execSteps = document.querySelectorAll('.exec-step');
  execSteps.forEach(step => {
    step.addEventListener('click', () => {
      const wasActive = step.classList.contains('active');
      execSteps.forEach(s => s.classList.remove('active'));
      if (!wasActive) {
        step.classList.add('active');
        playUiSound('click');
      }
    });
  });

  // ── Knowledge Gap Checkboxes ───────────────────────────────
  const gapCheckboxes = document.querySelectorAll('.gap-checkbox');
  const STORAGE_KEY = 'rag-playbook-gaps';
  let savedGaps = {};

  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) savedGaps = JSON.parse(saved);
  } catch (e) { /* ignore */ }

  gapCheckboxes.forEach(cb => {
    const gapId = cb.dataset.gapId;
    if (gapId && savedGaps[gapId]) {
      cb.checked = true;
      cb.closest('.gap-item')?.classList.add('checked');
    }

    cb.addEventListener('change', () => {
      const item = cb.closest('.gap-item');
      if (cb.checked) {
        item?.classList.add('checked');
        savedGaps[gapId] = true;
        playUiSound('success');
      } else {
        item?.classList.remove('checked');
        delete savedGaps[gapId];
        playUiSound('click');
      }
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(savedGaps));
      } catch (e) { /* ignore */ }
      updateGapCounter();
    });
  });

  function updateGapCounter() {
    const total = gapCheckboxes.length;
    const checked = document.querySelectorAll('.gap-checkbox:checked').length;
    const counter = document.querySelector('.gap-counter');
    if (counter) {
      counter.textContent = `${checked} / ${total} concepts mastered`;
    }
  }
  updateGapCounter();

  // ── Mobile Menu Toggle ─────────────────────────────────────
  const mobileMenuBtn = document.getElementById('mobileMenuBtn');
  const mobileDrawer = document.getElementById('mobileDrawer');
  const mobileLinks = document.querySelectorAll('.mobile-nav-link');

  if (mobileMenuBtn && mobileDrawer) {
    mobileMenuBtn.addEventListener('click', () => {
      mobileDrawer.classList.toggle('open');
      playUiSound('click');
    });

    mobileLinks.forEach(link => {
      link.addEventListener('click', () => {
        mobileDrawer.classList.remove('open');
      });
    });
  }

  // ── Escape Key Closes All ──────────────────────────────────
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && document.getElementById('cmdkModal')?.style.display !== 'flex') {
      questionBlocks.forEach(b => b.classList.remove('open'));
      followupItems.forEach(f => f.classList.remove('open'));
    }
  });

});

