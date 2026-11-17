import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import type { FileEntry, FolderView, StorageOverview, StorageReport } from "./api";

export function bytes(n: number): string {
  const units = ["B", "KB", "MB", "GB", "TB"];
  let i = 0;
  while (n >= 1024 && i < units.length - 1) { n /= 1024; i++; }
  return i < 2 ? `${Math.round(n)} ${units[i]}` : `${n.toFixed(1)} ${units[i]}`;
}

function duration(s: number): string {
  return s < 60 ? `${Math.round(s)} s` : `${Math.floor(s / 60)} min ${Math.round(s % 60)} s`;
}

function shortPath(path: string, root: string | undefined): string {
  if (root && path.toLowerCase().startsWith(root.toLowerCase())) return "~" + path.slice(root.length);
  return path;
}

const name = (path: string) => path.split(/[\\/]/).filter(Boolean).pop() ?? path;

/** Storage report. Look-only: every button here measures or shows, none changes a file. */
export default function Storage() {
  const [overview, setOverview] = useState<StorageOverview | null>(null);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(
    () => api.storage().then((o) => { setOverview(o); setError(null); }).catch((e) => setError(e.message)),
    [],
  );
  useEffect(() => { refresh(); }, [refresh]);

  const running = overview?.scan.state === "scanning" || overview?.scan.state === "duplicates";
  useEffect(() => {
    if (!running) return;
    const timer = setInterval(refresh, 1000);
    return () => clearInterval(timer);
  }, [running, refresh]);

  const report = overview?.report ?? null;

  return (
    <div className="storage">
      <header className="storage-head">
        <div>
          <h1>Storage</h1>
          <p className="muted">Look-only: Athena measures and explains. Nothing here changes or deletes a file.</p>
        </div>
      </header>
      {error && <p className="error">{error}</p>}
      {overview && <Drives overview={overview} />}
      {overview && <ScanCard overview={overview} onChange={refresh} />}
      {report && !running && <ReportView report={report} />}
    </div>
  );
}

function Drives({ overview }: { overview: StorageOverview }) {
  return (
    <section className="drives">
      {overview.drives.map((d) => (
        <div key={d.mount} className={`drive ${d.percent >= 90 ? "low" : ""}`}>
          <div className="drive-top"><strong>{d.mount}</strong><span>{bytes(d.free)} free</span></div>
          <div className="bar"><span style={{ width: `${d.percent}%` }} /></div>
          <small className="muted">{bytes(d.used)} used of {bytes(d.total)} · {d.percent.toFixed(0)}%</small>
        </div>
      ))}
      {overview.recycle_bin && (
        <div className="drive">
          <div className="drive-top"><strong>Recycle Bin</strong><span>{bytes(overview.recycle_bin.bytes)}</span></div>
          <small className="muted">{overview.recycle_bin.items} items. Emptying it comes in a later step.</small>
        </div>
      )}
    </section>
  );
}

function ScanCard({ overview, onChange }: { overview: StorageOverview; onChange: () => void }) {
  const { scan, report } = overview;
  const running = scan.state === "scanning" || scan.state === "duplicates";

  if (running) {
    const pct = scan.expected_files ? Math.min(99, (100 * (scan.files ?? 0)) / scan.expected_files) : null;
    return (
      <section className="card scan">
        <div className="scan-row">
          <strong>{scan.state === "duplicates" ? "Checking for duplicate files…" : "Scanning your home folder…"}</strong>
          <button onClick={() => api.cancelScan().then(onChange)}>Cancel</button>
        </div>
        <div className={`bar ${pct === null ? "indeterminate" : ""}`}><span style={{ width: `${pct ?? 30}%` }} /></div>
        <small className="muted">
          {(scan.files ?? 0).toLocaleString()} files · {bytes(scan.bytes ?? 0)}
          {scan.state === "duplicates" && ` · ${bytes(scan.hashed_bytes ?? 0)} compared`}
          {scan.elapsed_s != null && ` · ${duration(scan.elapsed_s)}`}
        </small>
        {scan.current && scan.state === "scanning" && <small className="muted current">{scan.current}</small>}
        <small className="muted">You can keep using Athena while this runs.</small>
      </section>
    );
  }

  return (
    <section className="card scan">
      <div className="scan-row">
        <div>
          {report ? (
            <>
              <strong>{bytes(report.bytes)} in {report.files.toLocaleString()} files</strong>
              <small className="muted block">
                Scanned {new Date(report.scanned_at * 1000).toLocaleString()} · took {duration(report.duration_s)}
                {report.unreadable_folders > 0 && ` · ${report.unreadable_folders} folders couldn't be read`}
              </small>
            </>
          ) : (
            <>
              <strong>See what's using your space</strong>
              <small className="muted block">
                Athena reads the names and sizes of the files in your home folder. A big folder can take a few minutes.
              </small>
            </>
          )}
        </div>
        <button className="primary" onClick={() => api.startScan().then(onChange)}>
          {report ? "Scan again" : "Scan"}
        </button>
      </div>
      {scan.state === "error" && <p className="error">Scan failed: {scan.error}</p>}
      {scan.state === "cancelled" && <small className="muted">Last scan was cancelled.</small>}
    </section>
  );
}

function Reveal({ path }: { path: string }) {
  const [failed, setFailed] = useState(false);
  return (
    <button className="reveal" title="Show in folder"
      onClick={() => api.reveal(path).catch(() => setFailed(true))}>
      {failed ? "Can't show" : "Show"}
    </button>
  );
}

function FileRow({ file, root }: { file: FileEntry; root: string }) {
  return (
    <li className="file-row">
      <div className="file-name">
        <span>{name(file.path)}</span>
        <small className="muted">{shortPath(file.path, root)}</small>
      </div>
      <span className="size">{bytes(file.bytes)}</span>
      <Reveal path={file.path} />
    </li>
  );
}

function ReportView({ report }: { report: StorageReport }) {
  const { root } = report;
  const [showAllLarge, setShowAllLarge] = useState(false);
  const dup = report.duplicates;
  const inst = report.old_installers;

  return (
    <>
      {report.space_hogs.length > 0 && (
        <section className="card">
          <h2>Space you might get back</h2>
          <ul className="hogs">
            {report.space_hogs.map((h) => (
              <li key={h.id}>
                <div className="hog-main">
                  <strong>{h.name}</strong>
                  <span className="size">{bytes(h.bytes)}</span>
                  <Reveal path={h.path} />
                </div>
                <small className="muted">{h.advice}</small>
              </li>
            ))}
          </ul>
        </section>
      )}

      <FolderExplorer root={root} />

      <section className="card">
        <h2>Largest files</h2>
        <ul className="files">
          {report.largest_files.slice(0, showAllLarge ? 50 : 10).map((f) => <FileRow key={f.path} file={f} root={root} />)}
        </ul>
        {!showAllLarge && report.largest_files.length > 10 && (
          <button className="more" onClick={() => setShowAllLarge(true)}>Show {report.largest_files.length - 10} more</button>
        )}
      </section>

      <section className="card">
        <h2>Old installers and archives in Downloads</h2>
        {inst.count === 0 ? (
          <p className="muted small">None older than {inst.days} days.</p>
        ) : (
          <>
            <p className="muted small">
              {inst.count} files older than {inst.days} days, {bytes(inst.bytes)} in total. Once a program is
              installed, its installer is usually not needed.
            </p>
            <ul className="files">{inst.files.slice(0, 15).map((f) => <FileRow key={f.path} file={f} root={root} />)}</ul>
          </>
        )}
      </section>

      <section className="card">
        <h2>Duplicate files</h2>
        {dup.count === 0 ? (
          <p className="muted small">No identical copies of your files larger than {bytes(dup.min_bytes)}.</p>
        ) : (
          <>
            <p className="muted small">
              {dup.count} sets of identical files (same content, checked byte by byte). Keeping one copy of each
              would free {bytes(dup.savings)}.
              {!dup.complete && " Athena stopped comparing after the biggest ones to keep the scan quick."}
            </p>
            <ul className="dups">
              {dup.groups.slice(0, 15).map((g) => (
                <li key={g.paths[0]}>
                  <div className="dup-head"><strong>{name(g.paths[0])}</strong>
                    <span className="muted small">{g.paths.length} copies × {bytes(g.bytes)}</span></div>
                  <ul className="files">
                    {g.paths.map((p) => <FileRow key={p} file={{ path: p, bytes: g.bytes }} root={root} />)}
                  </ul>
                </li>
              ))}
            </ul>
          </>
        )}
        <p className="hint">Only your own files are compared: copies inside app folders, caches and code
          dependencies are normal and left out.</p>
      </section>
    </>
  );
}

function FolderExplorer({ root }: { root: string }) {
  const [path, setPath] = useState(root);
  const [view, setView] = useState<FolderView | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let live = true;
    setError(null);
    api.folder(path).then((v) => live && setView(v)).catch((e) => live && setError(e.message));
    return () => { live = false; };
  }, [path]);

  const max = Math.max(1, ...(view?.folders.map((f) => f.bytes) ?? [1]));
  return (
    <section className="card">
      <h2>Folders</h2>
      {view && (
        <nav className="crumbs">
          {view.crumbs.map((c, i) => (
            <span key={c.path}>
              {i > 0 && <span className="muted"> / </span>}
              <button className="link" onClick={() => setPath(c.path)} disabled={i === view.crumbs.length - 1}>{c.name}</button>
            </span>
          ))}
          <span className="muted"> · {bytes(view.bytes)}</span>
        </nav>
      )}
      {error && <p className="error">{error}</p>}
      {view && (
        <>
          <ul className="folders">
            {view.folders.slice(0, 25).map((f) => (
              <li key={f.path}>
                <button className="folder" onClick={() => setPath(f.path)}>
                  <span className="folder-name">{name(f.path)}</span>
                  <span className="fbar"><span style={{ width: `${(100 * f.bytes) / max}%` }} /></span>
                  <span className="size">{bytes(f.bytes)}</span>
                </button>
                <Reveal path={f.path} />
              </li>
            ))}
          </ul>
          {view.small_folders.count > 0 && (
            <p className="muted small">+ {view.small_folders.count} smaller folders, {bytes(view.small_folders.bytes)} in total</p>
          )}
          {view.files.length > 0 && (
            <>
              <h3>Files in this folder ({view.file_count.toLocaleString()})</h3>
              <ul className="files">{view.files.slice(0, 15).map((f) => <FileRow key={f.path} file={f} root={root} />)}</ul>
            </>
          )}
        </>
      )}
    </section>
  );
}
