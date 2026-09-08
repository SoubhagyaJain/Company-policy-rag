// Kills any frontend dev server already running before `next dev` starts.
//
// Runs automatically as the npm `predev` hook. Next's dev server holds a native
// lockfile that also stores the running server's { pid, port } as JSON and
// refuses to start while another instance is alive ("Another next dev server is
// already running."). We read that lockfile for the PID and, as a fallback, find
// whatever is listening on the dev port, then kill it so a fresh `next dev` can
// always acquire the port and lock cleanly.
//
// It is deliberately best-effort: if nothing is running (or a PID is already
// dead), every step is a silent no-op — it must never fail the `dev` script.

import { execSync } from 'node:child_process';
import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';

const isWindows = process.platform === 'win32';
const port = Number(process.env.PORT) || 3000;
const cwd = process.cwd();

/** PIDs we must never kill: this script and the npm process that spawned it. */
const selfPids = new Set([process.pid, process.ppid].map(String));

/** Collect the PID Next recorded in its dev lockfile, if present. */
function pidsFromLockfiles() {
  const pids = new Set();
  // distDir defaults to `.next`; dev builds have also lived under `.next/dev`.
  for (const lockPath of [join(cwd, '.next', 'lock'), join(cwd, '.next', 'dev', 'lock')]) {
    if (!existsSync(lockPath)) continue;
    try {
      const info = JSON.parse(readFileSync(lockPath, 'utf-8'));
      if (info && info.pid) pids.add(String(info.pid));
    } catch {
      // Not JSON / unreadable — skip.
    }
  }
  return pids;
}

/** Collect PIDs currently LISTENING on the dev port. */
function pidsOnPort() {
  const pids = new Set();
  try {
    if (isWindows) {
      const out = execSync('netstat -ano -p tcp', { encoding: 'utf-8', stdio: ['ignore', 'pipe', 'ignore'] });
      for (const line of out.split(/\r?\n/)) {
        if (!/LISTENING/i.test(line)) continue;
        const cols = line.trim().split(/\s+/);
        const local = cols[1] || '';
        if (local.endsWith(`:${port}`)) {
          const pid = cols[cols.length - 1];
          if (pid && pid !== '0') pids.add(pid);
        }
      }
    } else {
      const out = execSync(`lsof -ti tcp:${port} -sTCP:LISTEN`, { encoding: 'utf-8', stdio: ['ignore', 'pipe', 'ignore'] });
      for (const pid of out.split(/\s+/)) if (pid) pids.add(pid.trim());
    }
  } catch {
    // Nothing listening (or tool unavailable) — nothing to collect.
  }
  return pids;
}

function kill(pid) {
  try {
    if (isWindows) execSync(`taskkill /PID ${pid} /F /T`, { stdio: 'ignore' });
    else process.kill(Number(pid), 'SIGKILL');
    console.log(`[kill-dev] stopped stale dev server (PID ${pid})`);
  } catch {
    // Already gone or not ours to kill — ignore.
  }
}

const targets = new Set([...pidsFromLockfiles(), ...pidsOnPort()]);
for (const pid of targets) {
  if (!selfPids.has(pid)) kill(pid);
}
