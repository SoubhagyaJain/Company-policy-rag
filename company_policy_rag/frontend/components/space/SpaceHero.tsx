'use client';

/**
 * SpaceHero — WebGL gravitational-lensing hero, ported from the Claude Design
 * "Space Hero.dc.html" Component class into a React effect. Behaviour preserved:
 * two-pass shader (scene + 5,200 particles), pointer parallax, scroll-linked
 * intensification, in-shader light/dark cross-fade (uLight), adaptive DPR
 * degradation, prefers-reduced-motion, mobile tier, and the CSS heroBreathe
 * fallback when WebGL / shader compilation is unavailable.
 *
 * The texture is loaded via `new Image()` (not next/image) because it is sampled
 * into a GL texture, not painted to the DOM.
 */

import { useEffect, useRef } from 'react';

const VERT_SCENE = `
attribute vec2 aPos;
varying vec2 vUv;
void main(){ vUv = aPos * 0.5 + 0.5; gl_Position = vec4(aPos, 0.0, 1.0); }`;

const FRAG_SCENE = `
precision highp float;
varying vec2 vUv;
uniform sampler2D uTex;
uniform vec2 uRes;
uniform vec2 uMouse;
uniform float uImgA, uTime, uScroll, uBloom, uLens, uPar, uFade, uLight;

const vec2 C = vec2(0.483, 0.597);

float hash(vec2 p){ return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }
float noise(vec2 p){
  vec2 i = floor(p), f = fract(p);
  f = f * f * (3.0 - 2.0 * f);
  return mix(mix(hash(i), hash(i + vec2(1.0, 0.0)), f.x),
             mix(hash(i + vec2(0.0, 1.0)), hash(i + vec2(1.0, 1.0)), f.x), f.y);
}
float fbm(vec2 p){ return noise(p) * 0.6 + noise(p * 2.13) * 0.28 + noise(p * 4.31) * 0.12; }
float lumf(vec3 c){ return dot(c, vec3(0.299, 0.587, 0.114)); }

mat3 camRot(float yaw, float pitch){
  float cy = cos(yaw), sy = sin(yaw), cp = cos(pitch), sp = sin(pitch);
  return mat3(cy, 0.0, -sy, 0.0, 1.0, 0.0, sy, 0.0, cy)
       * mat3(1.0, 0.0, 0.0, 0.0, cp, sp, 0.0, -sp, cp);
}

vec2 planeUv(vec3 ro, vec3 rd, float d){
  float t = (d - ro.z) / max(rd.z, 0.05);
  return (ro.xy + rd.xy * t) / d + 0.5;
}

vec2 warp(vec2 uv, float lensAmt, float swirlAmt, float turb){
  vec2 q = vec2(uv.x - C.x, (uv.y - C.y) / uImgA);
  float r = length(q), a = atan(q.y, q.x);
  float ann = exp(-pow((r - 0.132) / 0.088, 2.0));
  vec2 lq = q * (1.0 - (lensAmt * exp(-r * 7.0)) / max(r, 0.022));
  float sw = (uTime * 0.075 + fbm(vec2(a * 1.7, r * 11.0 - uTime * 0.28)) * 0.75) * ann * swirlAmt;
  float cs = cos(sw), sn = sin(sw);
  lq = mat2(cs, sn, -sn, cs) * lq;
  lq += (vec2(fbm(lq * 10.0 + uTime * 0.14), fbm(lq * 10.0 + 7.31 - uTime * 0.12)) - 0.5) * turb * ann;
  lq += (vec2(fbm(lq * 3.2 - uTime * 0.035), fbm(lq * 3.2 + 4.1 + uTime * 0.03)) - 0.5) * 0.006;
  return clamp(vec2(lq.x + C.x, lq.y * uImgA + C.y), 0.002, 0.998);
}

void main(){
  float sa = uRes.x / max(uRes.y, 1.0);
  vec2 p = vUv - 0.5;
  if (sa > uImgA) p.y *= uImgA / sa; else p.x *= sa / uImgA;

  vec2 drift = vec2(sin(uTime * 0.061) * 0.011 + sin(uTime * 0.137 + 1.3) * 0.004,
                    cos(uTime * 0.049) * 0.008 + cos(uTime * 0.113 + 0.7) * 0.003);

  float yaw   = (uMouse.x * 0.135 + sin(uTime * 0.061) * 0.016) * uPar;
  float pitch = (uMouse.y * 0.095 + cos(uTime * 0.049) * 0.013) * uPar;
  vec3 ro = vec3(uMouse.x * 0.090 * uPar + drift.x, uMouse.y * 0.062 * uPar + drift.y,
                 uScroll * 0.42);
  vec3 rd = camRot(yaw, pitch) * normalize(vec3(p, 1.0));

  vec2 uvF = planeUv(ro, rd, 3.10);
  vec2 uvM = planeUv(ro, rd, 1.00);
  vec2 uvN = planeUv(ro, rd, 0.44);

  vec2 N = normalize(vec2(-0.17, 1.0));
  float lensK = uLens * (0.020 + 0.005 * sin(uTime * 0.23)) * (1.0 + uScroll * 0.55)
              * (1.0 + 0.10 * uMouse.x * uPar);

  vec2 wF = warp(uvF, lensK * 1.4, 0.07, 0.007);
  vec2 wM = warp(uvM, lensK, 0.20, 0.015);

  vec2 q = vec2(uvM.x - C.x, (uvM.y - C.y) / uImgA);
  float r = length(q);
  float a = atan(q.y, q.x);
  float annulus = exp(-pow((r - 0.132) / 0.088, 2.0));

  vec3 col = texture2D(uTex, wF).rgb;
  col *= 1.0 - 0.80 * smoothstep(0.04, -0.24, dot(uvF - C, N));

  vec3 cm = texture2D(uTex, wM).rgb;
  float glowM = smoothstep(0.20, 0.62, lumf(cm));
  float disp = 0.0028 * annulus * uLens;
  cm.r = texture2D(uTex, clamp(wM + q * disp, 0.002, 0.998)).r;
  cm.b = texture2D(uTex, clamp(wM - q * disp, 0.002, 0.998)).b;
  col = mix(col, cm, glowM * 0.92);

  vec3 cn = texture2D(uTex, clamp(uvN, 0.002, 0.998)).rgb;
  float nA = smoothstep(0.05, -0.26, dot(uvN - C, N));
  col = mix(col, cn, nA * 0.94);

  vec3 bl = vec3(0.0);
  for (int i = 0; i < 8; i++){
    float ang = float(i) * 0.7853981 + uTime * 0.02;
    vec2 d = vec2(cos(ang), sin(ang)) * 0.014;
    bl += max(texture2D(uTex, clamp(wM + d, 0.002, 0.998)).rgb - 0.34, 0.0);
  }
  col += (bl / 8.0) * uBloom * 1.6 * (1.0 + uScroll * 0.9);

  float flick = 0.80 + 0.20 * fbm(vec2(a * 3.0 - uTime * 0.35, uTime * 0.9));
  float pulse = 0.9 + 0.10 * sin(uTime * 1.7 + a * 2.0);
  col *= 1.0 + annulus * glowM * (0.30 * flick * pulse + uScroll * 0.60);
  col += vec3(0.40, 0.66, 1.0) * exp(-r * 6.4) * 0.16
       * (0.86 + 0.14 * sin(uTime * 0.55) + 0.05 * sin(uTime * 1.9)) * (1.0 + uScroll * 0.7);
  col += vec3(0.55, 0.10, 0.24) * exp(-length(vec2((uvN.x - 0.50) * 1.6, (uvN.y - 0.36) / uImgA)) * 7.0)
       * 0.055 * (0.85 + 0.15 * sin(uTime * 0.31 + 1.7));
  col *= 1.0 - 0.40 * smoothstep(0.095, 0.030, r);

  vec2 v = (vUv - 0.5) * vec2(sa, 1.0);
  col *= 1.0 - 0.30 * dot(v, v);
  col = mix(vec3(0.014, 0.019, 0.036), col, uFade);

  if (uLight > 0.001){
    float l = lumf(col);
    float ink = pow(clamp(l * 1.22, 0.0, 1.0), 0.82);
    vec3 paper = vec3(0.964, 0.968, 0.980);
    vec3 inkC = vec3(0.075, 0.105, 0.185);
    vec3 lightCol = mix(paper, inkC, ink);
    lightCol += (col - vec3(l)) * 0.30 * (1.0 - ink);
    lightCol = mix(lightCol, paper, 0.10);
    col = mix(col, clamp(lightCol, 0.0, 1.0), uLight);
  }
  gl_FragColor = vec4(col, 1.0);
}`;

const VERT_PART = `
precision highp float;
attribute vec3 aSeed;
attribute float aKind;
uniform vec2 uRes, uMouse;
uniform float uTime, uScroll, uPar, uDpr, uFade, uLight;
varying float vA, vK;

mat3 camRotInv(float yaw, float pitch){
  float cy = cos(yaw), sy = sin(yaw), cp = cos(pitch), sp = sin(pitch);
  return mat3(1.0, 0.0, 0.0, 0.0, cp, -sp, 0.0, sp, cp)
       * mat3(cy, 0.0, sy, 0.0, 1.0, 0.0, -sy, 0.0, cy);
}

void main(){
  float s = aSeed.z;
  float ph = aSeed.x * 31.4 + aSeed.y * 17.7;
  float sa = uRes.x / max(uRes.y, 1.0);
  float span = 6.2;

  float cyc = fract(ph * 0.081 + uTime * (0.020 + s * 0.085) + uScroll * 0.55);
  float z = 0.12 + (1.0 - cyc) * span;

  float spread = 1.15 + s * 0.6 + z * 0.30;
  vec3 wp = vec3((aSeed.xy * 2.0 - 1.0) * spread, z);
  wp.xy += vec2(sin(uTime * (0.09 + s * 0.22) + ph), cos(uTime * (0.075 + s * 0.19) + ph * 1.3))
           * (0.05 + s * 0.13);

  float orb = uTime * (0.030 + s * 0.055) / (0.45 + z * 0.55);
  float co = cos(orb), so = sin(orb);
  wp.xy = mat2(co, -so, so, co) * wp.xy;

  float yaw   = (uMouse.x * 0.135 + sin(uTime * 0.061) * 0.016) * uPar;
  float pitch = (uMouse.y * 0.095 + cos(uTime * 0.049) * 0.013) * uPar;
  vec3 ro = vec3(uMouse.x * 0.090 * uPar, uMouse.y * 0.062 * uPar, 0.0);

  vec3 v = camRotInv(yaw, pitch) * (wp - ro);
  vec2 ndc = vec2(v.x / max(v.z, 0.06) / sa, v.y / max(v.z, 0.06));

  float tw = 0.55 + 0.45 * sin(uTime * (1.3 + s * 4.0) + ph * 4.0);
  float fadeNear = smoothstep(0.12, 0.48, z);
  float fadeFar = 1.0 - smoothstep(span * 0.66, span + 0.2, z);
  float prox = clamp(1.6 / z, 0.0, 3.0);
  vA = (0.24 + s * 0.72) * tw * fadeNear * fadeFar * (0.55 + prox * 0.45)
       * uFade * (1.0 - uLight * 0.55);
  vK = aKind;
  gl_PointSize = (0.7 + s * 2.2 + aKind * 1.7) * uDpr * clamp(1.9 / z, 0.35, 9.0)
                 * (1.0 + uScroll * 0.5);
  gl_Position = vec4(ndc, 0.0, 1.0);
}`;

const FRAG_PART = `
precision highp float;
uniform float uLight;
varying float vA, vK;
void main(){
  vec2 d = gl_PointCoord - 0.5;
  float m = smoothstep(0.5, 0.03, length(d));
  vec3 c = mix(vec3(0.80, 0.90, 1.0), vec3(1.0, 0.34, 0.52), step(0.5, vK));
  c = mix(c, vec3(1.0), step(1.5, vK) * 0.7);
  c = mix(c, vec3(0.16, 0.22, 0.36), uLight);
  gl_FragColor = vec4(c * m * vA, m * vA);
}`;

export interface SpaceHeroProps {
  /** When true the shader cross-fades to the light "astronomical negative" palette. */
  light?: boolean;
  /** The scroll runway element (e.g. the 220vh outer div); drives shader intensity. */
  runwayRef?: React.RefObject<HTMLElement | null>;
  parallax?: number;
  bloom?: number;
  lensing?: number;
  particleDensity?: number;
  imageSrc?: string;
  className?: string;
}

export function SpaceHero({
  light = true,
  runwayRef,
  parallax = 1,
  bloom = 0.8,
  lensing = 1,
  particleDensity = 1,
  imageSrc = '/space-hero.jpg',
  className,
}: SpaceHeroProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const lightRef = useRef(light);
  const propsRef = useRef({ parallax, bloom, lensing, particleDensity });

  // Keep live values readable by the running frame loop without restarting it.
  useEffect(() => {
    lightRef.current = light;
  }, [light]);
  useEffect(() => {
    propsRef.current = { parallax, bloom, lensing, particleDensity };
  }, [parallax, bloom, lensing, particleDensity]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const small = window.matchMedia('(max-width: 820px)').matches;
    const dpr = Math.min(window.devicePixelRatio || 1, small ? 1.5 : 2);

    const mouse = { x: 0, y: 0, tx: 0, ty: 0 };
    let scroll = 0;
    let scrollT = 0;
    let t = 0;
    let fade = 0;
    let lightMix = light ? 1 : 0;
    let frames = 0;
    let acc = 0;
    let degraded = false;
    let last = 0;
    let raf = 0;

    const smoothDamp = (current: number, target: number, halfLife: number, dt: number) => {
      const factor = 1 - Math.pow(2, -dt / halfLife);
      return current + (target - current) * factor;
    };

    let gl: WebGLRenderingContext | null = null;
    let progScene: WebGLProgram | null = null;
    let progPart: WebGLProgram | null = null;
    let quad: WebGLBuffer | null = null;
    let pbuf: WebGLBuffer | null = null;
    let tex: WebGLTexture | null = null;
    let uS: Record<string, WebGLUniformLocation | null> = {};
    let uP: Record<string, WebGLUniformLocation | null> = {};
    let aPos = 0;
    let aSeed = 0;
    let aKind = 0;
    let imgA = 1200 / 558;
    let count = 0;

    const fallback = () => {
      const im = imgRef.current;
      if (im && !reduced) im.style.animation = 'heroBreathe 26s ease-in-out infinite';
    };

    const compile = (vs: string, fs: string): WebGLProgram => {
      const g = gl!;
      const mk = (type: number, src: string) => {
        const sh = g.createShader(type)!;
        g.shaderSource(sh, src);
        g.compileShader(sh);
        if (!g.getShaderParameter(sh, g.COMPILE_STATUS)) throw new Error(g.getShaderInfoLog(sh) || 'compile');
        return sh;
      };
      const p = g.createProgram()!;
      g.attachShader(p, mk(g.VERTEX_SHADER, vs));
      g.attachShader(p, mk(g.FRAGMENT_SHADER, fs));
      g.linkProgram(p);
      if (!g.getProgramParameter(p, g.LINK_STATUS)) throw new Error(g.getProgramInfoLog(p) || 'link');
      return p;
    };

    const uni = (prog: WebGLProgram, names: string[]) => {
      const g = gl!;
      const o: Record<string, WebGLUniformLocation | null> = {};
      names.forEach((n) => { o[n] = g.getUniformLocation(prog, n); });
      return o;
    };

    const resize = () => {
      if (!gl || !canvas) return;
      const w = Math.max(canvas.clientWidth, 1);
      const h = Math.max(canvas.clientHeight, 1);
      const d = degraded ? 1 : dpr;
      canvas.width = Math.round(w * d);
      canvas.height = Math.round(h * d);
      gl.viewport(0, 0, canvas.width, canvas.height);
    };

    const onScroll = () => {
      const runway = runwayRef?.current;
      if (!runway) { scrollT = 0; return; }
      const rect = runway.getBoundingClientRect();
      const len = Math.max(rect.height - window.innerHeight, 1);
      scrollT = Math.min(Math.max(-rect.top / len, 0), 1);
    };

    const onMove = (e: PointerEvent) => {
      if (reduced) return;
      mouse.tx = (e.clientX / window.innerWidth) * 2 - 1;
      mouse.ty = -((e.clientY / window.innerHeight) * 2 - 1);
    };

    const onResize = () => { resize(); onScroll(); };

    const frame = (now: number) => {
      if (!gl) return;
      raf = requestAnimationFrame(frame);
      const dt = Math.min((now - last) / 1000, 0.05);
      last = now;

      if (!degraded) {
        acc += dt; frames++;
        if (frames === 90) {
          if (acc / 90 > 0.024) { degraded = true; resize(); }
          frames = 0; acc = 0;
        }
      }

      const fadeTarget = 1;
      fade = smoothDamp(fade, fadeTarget, 0.32, dt);
      if (fadeTarget - fade < 0.002) fade = 1;
      if (!reduced) t += dt;
      mouse.x = smoothDamp(mouse.x, mouse.tx, 0.12, dt);
      mouse.y = smoothDamp(mouse.y, mouse.ty, 0.12, dt);
      scroll = smoothDamp(scroll, scrollT, 0.18, dt);
      const lt = lightRef.current ? 1 : 0;
      lightMix = smoothDamp(lightMix, lt, 0.22, dt);

      const cfg = propsRef.current;
      const par = (reduced ? 0 : cfg.parallax) * (small ? 0.45 : 1);
      const bl = cfg.bloom * (degraded || small ? 0.6 : 1);
      const lens = cfg.lensing * (degraded ? 0.4 : 1);
      const cv = canvas;

      gl.useProgram(progScene!);
      gl.blendFunc(gl.ONE, gl.ZERO);
      gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.uniform1i(uS.uTex, 0);
      gl.uniform2f(uS.uRes, cv.width, cv.height);
      gl.uniform2f(uS.uMouse, mouse.x, mouse.y);
      gl.uniform1f(uS.uImgA, imgA);
      gl.uniform1f(uS.uTime, t);
      gl.uniform1f(uS.uScroll, scroll);
      gl.uniform1f(uS.uBloom, bl);
      gl.uniform1f(uS.uLens, lens);
      gl.uniform1f(uS.uPar, par);
      gl.uniform1f(uS.uFade, fade);
      gl.uniform1f(uS.uLight, lightMix);
      gl.drawArrays(gl.TRIANGLES, 0, 3);

      gl.useProgram(progPart!);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE);
      gl.bindBuffer(gl.ARRAY_BUFFER, pbuf);
      gl.enableVertexAttribArray(aSeed);
      gl.vertexAttribPointer(aSeed, 3, gl.FLOAT, false, 16, 0);
      gl.enableVertexAttribArray(aKind);
      gl.vertexAttribPointer(aKind, 1, gl.FLOAT, false, 16, 12);
      gl.uniform2f(uP.uRes, cv.width, cv.height);
      gl.uniform2f(uP.uMouse, mouse.x, mouse.y);
      gl.uniform1f(uP.uTime, t);
      gl.uniform1f(uP.uScroll, scroll);
      gl.uniform1f(uP.uPar, par);
      gl.uniform1f(uP.uDpr, degraded ? 1 : dpr);
      gl.uniform1f(uP.uFade, Math.max(fade * 1.6 - 0.6, 0));
      gl.uniform1f(uP.uLight, lightMix);
      gl.drawArrays(gl.POINTS, 0, count);
    };

    const initGL = () => {
      gl = canvas.getContext('webgl', {
        alpha: false, antialias: false, depth: false, stencil: false,
        premultipliedAlpha: true, powerPreference: 'high-performance',
      }) as WebGLRenderingContext | null;
      if (!gl) return fallback();

      try {
        progScene = compile(VERT_SCENE, FRAG_SCENE);
        progPart = compile(VERT_PART, FRAG_PART);
      } catch (err) {
        console.warn('SpaceHero shader compile failed', err);
        return fallback();
      }

      uS = uni(progScene, ['uTex', 'uRes', 'uMouse', 'uImgA', 'uTime', 'uScroll', 'uBloom', 'uLens', 'uPar', 'uFade', 'uLight']);
      uP = uni(progPart, ['uRes', 'uMouse', 'uTime', 'uScroll', 'uPar', 'uDpr', 'uFade', 'uLight']);

      quad = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, quad);
      gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 3, -1, -1, 3]), gl.STATIC_DRAW);
      aPos = gl.getAttribLocation(progScene, 'aPos');

      const dens = propsRef.current.particleDensity ?? 1;
      count = Math.round((small ? 1400 : 5200) * dens * (reduced ? 0.3 : 1));
      const data = new Float32Array(count * 4);
      for (let i = 0; i < count; i++) {
        const rnd = Math.random();
        data[i * 4 + 0] = Math.random();
        data[i * 4 + 1] = Math.random();
        data[i * 4 + 2] = Math.pow(Math.random(), 1.9);
        data[i * 4 + 3] = rnd > 0.985 ? 2 : rnd > 0.955 ? 1 : 0;
      }
      pbuf = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, pbuf);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      aSeed = gl.getAttribLocation(progPart, 'aSeed');
      aKind = gl.getAttribLocation(progPart, 'aKind');

      tex = gl.createTexture();
      gl.bindTexture(gl.TEXTURE_2D, tex);
      gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
      gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGB, gl.RGB, gl.UNSIGNED_BYTE, img);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
      imgA = (img.naturalWidth || 1200) / (img.naturalHeight || 558);

      gl.disable(gl.DEPTH_TEST);
      gl.enable(gl.BLEND);
      resize();
      canvas.style.opacity = '1';
      last = performance.now();
      raf = requestAnimationFrame(frame);
    };

    const img = new Image();
    img.decoding = 'async';
    img.onload = () => initGL();
    img.onerror = () => fallback();
    img.src = imageSrc;

    window.addEventListener('pointermove', onMove, { passive: true });
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onResize);
    onScroll();

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onResize);
      if (gl) {
        [progScene, progPart].forEach((p) => p && gl!.deleteProgram(p));
        [quad, pbuf].forEach((b) => b && gl!.deleteBuffer(b));
        if (tex) gl.deleteTexture(tex);
        const lose = gl.getExtension('WEBGL_lose_context');
        if (lose) lose.loseContext();
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [imageSrc, runwayRef]);

  return (
    <div className={className} aria-hidden="true">
      <img
        ref={imgRef}
        src={imageSrc}
        alt=""
        aria-hidden="true"
        style={{
          position: 'absolute', inset: 0, width: '100%', height: '100%',
          objectFit: 'cover', transformOrigin: '48% 40%', willChange: 'transform',
        }}
      />
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute', inset: 0, width: '100%', height: '100%',
          display: 'block', opacity: 0, transition: 'opacity 1.2s cubic-bezier(0.22, 1, 0.36, 1)',
        }}
      />
    </div>
  );
}

export default SpaceHero;
