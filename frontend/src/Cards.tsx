// Result cards for the laptop assistant: what each tool found, as something
// to look at and click, not a wall of text. Every button here is read-only:
// Show (select in File Explorer), Dig in (read a file), Open (look inside a
// folder), Scan (measure the home folder).
import { useState } from "react";
import { api } from "./api";
import { bytes } from "./Storage";

export type Action = { tool: "dig" | "folder"; path: string; question?: string };
export type OnAction = (label: string, action?: Action) => void;

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Card = Record<string, any>;

const base = (p: string) => p.split(/[\\/]/).filter(Boolean).pop() ?? p;

function ago(ts?: number) {
  if (!ts) return "";
  const d = (Date.now() / 1000 - ts) / 86400;
  if (d < 1) return "today";
  if (d < 2) return "yesterday";
  if (d < 31) return `${Math.floor(d)} days ago`;
  if (d < 365) return `${Math.floor(d / 30)} months ago`;
  return `${Math.floor(d / 365)} years ago`;
}

const ICONS: Record<string, string> = {
  storage: "M3 7h18v10H3zM7 12h.01M11 12h6", folder: "M3 6h6l2 2h10v10H3z", files: "M10 4a6 6 0 104.47 10l5 5 1.5-1.5-5-5A6 6 0 0010 4z",
  file: "M6 3h8l4 4v14H6zM14 3v4h4", system: "M4 5h16v11H4zM9 20h6M12 16v4", security: "M12 3l8 3v6c0 5-3.5 8-8 9-4.5-1-8-4-8-9V6z",
  cleanup: "M4 20l6-6M14 4l6 6-8 8-6-6zM9 9l6 6", help: "M12 3a9 9 0 110 18 9 9 0 010-18zM9.5 9a2.5 2.5 0 015 .5c0 2-2.5 2-2.5 4M12 17h.01",
  choose: "M4 6h16M4 12h10M4 18h7",
};

function Head({ kind, title, sub, tone }: { kind: string; title: string; sub?: string; tone?: string }) {
  return (
    <header className="card-head">
      <span className={`card-icon ${tone ?? kind}`}>
        <svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" strokeWidth="1.8"
          strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={ICONS[kind] ?? ICONS.help} /></svg>
      </span>
      <div><strong>{title}</strong>{sub && <small>{sub}</small>}</div>
    </header>
  );
}

function Show({ path }: { path: string }) {
  const [state, setState] = useState<"idle" | "done" | "fail">("idle");
  return (
    <button className="mini" title="Show in File Explorer"
      onClick={() => api.reveal(path).then(() => setState("done")).catch(() => setState("fail"))}>
      {state === "fail" ? "Can't show" : state === "done" ? "Shown" : "Show"}
    </button>
  );
}

function Bar({ value, max, tone }: { value: number; max: number; tone?: string }) {
  const pct = Math.max(2, Math.min(100, (100 * value) / Math.max(1, max)));
  return <span className={`hbar ${tone ?? ""}`}><span style={{ width: `${pct}%` }} /></span>;
}

function Gauge({ label, pct, text, tone }: { label: string; pct: number; text: string; tone?: string }) {
  const t = tone ?? (pct >= 90 ? "bad" : pct >= 70 ? "warn" : "good");
  return (
    <div className={`gauge ${t}`} style={{ ["--p" as string]: `${pct}` }}>
      <div className="ring"><span>{Math.round(pct)}%</span></div>
      <strong>{label}</strong><small>{text}</small>
    </div>
  );
}

export default function ResultCard({ card, onAction }: { card: Card; onAction: OnAction }) {
  switch (card.kind) {
    case "storage": return <StorageCard card={card} onAction={onAction} />;
    case "folder": return <FolderCard card={card} onAction={onAction} />;
    case "files": return <FilesCard card={card} onAction={onAction} />;
    case "choose": return <FilesCard card={card} onAction={onAction} choose />;
    case "file": return <FileCard card={card} />;
    case "system": return <SystemCard card={card} />;
    case "security": return <SecurityCard card={card} />;
    case "cleanup": return <CleanupCard card={card} onAction={onAction} />;
    case "help": return <HelpCard card={card} onAction={onAction} />;
    default: return null;
  }
}

function StorageCard({ card, onAction }: { card: Card; onAction: OnAction }) {
  const [scanning, setScanning] = useState(false);
  const maxFolder = Math.max(1, ...(card.folders ?? []).map((f: Card) => f.bytes));
  return (
    <section className="rcard">
      <Head kind="storage" title="Storage" sub={card.scanned_at ? `Folders measured ${ago(card.scanned_at)}` : undefined} />
      <div className="drive-row">
        {card.drives.map((d: Card) => (
          <div key={d.mount} className={`drive-pill ${d.percent >= 90 ? "low" : ""}`}>
            <div className="dp-top"><strong>{d.mount}</strong><span>{bytes(d.free)} free</span></div>
            <Bar value={d.used} max={d.total} tone={d.percent >= 90 ? "bad" : ""} />
            <small>{d.percent.toFixed(0)}% of {bytes(d.total)} used</small>
          </div>
        ))}
        {card.recycle_bin?.bytes > 0 && (
          <div className="drive-pill bin"><div className="dp-top"><strong>Recycle Bin</strong>
            <span>{bytes(card.recycle_bin.bytes)}</span></div><small>{card.recycle_bin.items} items you deleted</small></div>
        )}
      </div>
      {card.folders ? (
        <>
          <h4>Biggest folders</h4>
          <ul className="rows">
            {card.folders.map((f: Card) => (
              <li key={f.path}>
                <button className="row-main" onClick={() => onAction(`What's in ${base(f.path)}?`, { tool: "folder", path: f.path })}>
                  <span className="row-name">{base(f.path)}{f.label && <small>{f.label}</small>}</span>
                  <Bar value={f.bytes} max={maxFolder} /><span className="size">{bytes(f.bytes)}</span>
                </button>
                <Show path={f.path} />
              </li>
            ))}
          </ul>
          {card.space_hogs?.length > 0 && (
            <>
              <h4>Space you could win back</h4>
              <div className="chips-row">
                {card.space_hogs.map((h: Card) => (
                  <span key={h.id} className="hog-chip" title={h.advice}><strong>{bytes(h.bytes)}</strong> {h.name}</span>
                ))}
              </div>
            </>
          )}
        </>
      ) : (
        <div className="cta">
          <span>Measure your folders to see what's using the space (a few minutes, read-only).</span>
          <button className="primary" disabled={scanning}
            onClick={() => { setScanning(true); api.startScan().then(() => setTimeout(() => onAction("How's my storage?"), 1500)); }}>
            {scanning ? "Scanning…" : "Scan now"}
          </button>
        </div>
      )}
    </section>
  );
}

function FolderCard({ card, onAction }: { card: Card; onAction: OnAction }) {
  const max = Math.max(1, ...card.folders.map((f: Card) => f.bytes));
  return (
    <section className="rcard">
      <Head kind="folder" title={card.display} sub={`${bytes(card.bytes)} · ${card.files.toLocaleString()} files${card.label ? ` · ${card.label}` : ""}`} />
      <nav className="crumbs small">
        {card.crumbs.map((c: Card, i: number) => (
          <span key={c.path}>{i > 0 && <span className="muted"> › </span>}
            <button className="link" disabled={i === card.crumbs.length - 1}
              onClick={() => onAction(`What's in ${c.name === "~" ? "my home folder" : c.name}?`, { tool: "folder", path: c.path })}>{c.name}</button>
          </span>
        ))}
        <span className="grow" /><Show path={card.path} />
      </nav>
      <ul className="rows">
        {card.folders.map((f: Card) => (
          <li key={f.path}>
            <button className="row-main" onClick={() => onAction(`What's in ${base(f.path)}?`, { tool: "folder", path: f.path })}>
              <span className="row-name">{base(f.path)}{f.label && <small>{f.label}</small>}</span>
              <Bar value={f.bytes} max={max} /><span className="size">{bytes(f.bytes)}</span>
            </button>
            <Show path={f.path} />
          </li>
        ))}
      </ul>
      {card.small_folders?.count > 0 && <p className="muted small">+ {card.small_folders.count} smaller folders, {bytes(card.small_folders.bytes)}</p>}
      {card.largest_files?.length > 0 && (
        <>
          <h4>Largest files inside</h4>
          <ul className="rows files">
            {card.largest_files.map((f: Card) => (
              <li key={f.path}><span className="row-name">{base(f.path)}</span><span className="size">{bytes(f.bytes)}</span><Show path={f.path} /></li>
            ))}
          </ul>
        </>
      )}
      {card.space_hogs?.length > 0 && (
        <div className="chips-row">
          {card.space_hogs.map((h: Card) => <span key={h.id} className="hog-chip" title={h.advice}><strong>{bytes(h.bytes)}</strong> {h.name}</span>)}
        </div>
      )}
      {card.protected && <p className="note">🔒 Athena won't delete things here directly: {card.protected}.</p>}
    </section>
  );
}

const KIND_LABEL: Record<string, string> = { pdf: "PDF", document: "DOC", spreadsheet: "XLS", presentation: "PPT", image: "IMG",
  video: "VID", audio: "AUD", archive: "ZIP", installer: "APP", code: "CODE", other: "FILE" };

function FilesCard({ card, onAction, choose }: { card: Card; onAction: OnAction; choose?: boolean }) {
  const results: Card[] = card.results ?? [];
  const readable = (r: Card) => ["pdf", "document", "presentation", "code"].includes(r.kind) || /\.(txt|md|csv|json|html?)$/i.test(r.name);
  const sub = choose ? "Several files match: pick one to read"
    : `${card.total} match${card.total === 1 ? "" : "es"} in ${card.searched?.toLocaleString?.() ?? "your"} files${card.kind_relaxed ? " (included other file types)" : ""}`;
  return (
    <section className="rcard">
      <Head kind={choose ? "choose" : "files"} title={choose ? "Which file?" : "Files"} sub={sub} />
      <ol className="file-list">
        {results.map((r, i) => (
          <li key={r.path}>
            <span className="idx">{i + 1}</span>
            <span className={`ftype ${r.kind}`}>{KIND_LABEL[r.kind] ?? "FILE"}</span>
            <span className="fmeta">
              <strong title={r.path}>{r.name}</strong>
              <small>{r.folder} · {bytes(r.bytes)} · {ago(r.modified)}{r.match?.includes("content") ? " · matched what's inside" : ""}
                {r.online_only ? " · online only" : ""}</small>
            </span>
            {readable(r) && (
              <button className="mini accent" onClick={() => onAction(`Dig into ${r.name}`, { tool: "dig", path: r.path, question: card.question ?? "summary" })}>
                Dig in
              </button>
            )}
            <Show path={r.path} />
          </li>
        ))}
      </ol>
      {!choose && card.content_index && !card.content_index.ready && card.content_index.total > 0 && (
        <p className="muted small">Still reading your documents ({card.content_index.done}/{card.content_index.total}); matches by content will improve.</p>
      )}
    </section>
  );
}

function FileCard({ card }: { card: Card }) {
  return (
    <section className="rcard compact">
      <Head kind="file" title={card.name}
        sub={[card.folder, bytes(card.bytes), card.pages ? `${card.pages} pages` : null, `modified ${ago(card.modified)}`].filter(Boolean).join(" · ")} />
      <div className="row-actions"><Show path={card.path} /></div>
    </section>
  );
}

function SystemCard({ card }: { card: Card }) {
  const m = card.memory, c = card.cpu, b = card.battery;
  const maxApp = Math.max(1, ...card.top_apps.map((a: Card) => a.bytes));
  const hours = Math.floor(card.uptime_s / 3600);
  return (
    <section className="rcard">
      <Head kind="system" title="Your laptop right now" sub={`On for ${hours >= 24 ? `${Math.floor(hours / 24)} days ${hours % 24} h` : `${hours} h`} since restart`} />
      <div className="gauges">
        <Gauge label="Memory" pct={m.percent} text={`${bytes(m.used)} of ${bytes(m.total)}`} />
        <Gauge label="CPU" pct={c.percent} text={`${c.cores} threads${c.ghz ? ` · ${c.ghz} GHz` : ""}`} />
        {b && <Gauge label="Battery" pct={b.percent} tone={b.percent <= 20 && !b.plugged_in ? "bad" : "good"}
          text={b.plugged_in ? "Charging" : b.minutes_left ? `${Math.floor(b.minutes_left / 60)} h ${b.minutes_left % 60} m left` : "On battery"} />}
      </div>
      <h4>Using the most memory</h4>
      <ul className="rows">
        {card.top_apps.map((a: Card) => (
          <li key={a.name}><span className="row-name">{a.name}{a.processes > 1 && <small>{a.processes} processes</small>}</span>
            <Bar value={a.bytes} max={maxApp} /><span className="size">{bytes(a.bytes)}</span></li>
        ))}
      </ul>
    </section>
  );
}

function SecurityCard({ card }: { card: Card }) {
  const tone = card.verdict === "danger" ? "bad" : card.verdict === "check" ? "warn" : "good";
  const title = card.verdict === "danger" ? "Signs of malware found" : card.verdict === "check" ? "A few things to check"
    : card.verdict === "unsupported" ? "Only available on Windows" : "No signs of malware";
  const c = card.counts ?? {};
  return (
    <section className={`rcard security ${tone}`}>
      <Head kind="security" tone={tone} title={title}
        sub={c.tasks != null ? `Checked ${c.tasks} scheduled tasks · ${c.autostart} startup items · ${c.processes} programs · ${c.connections} connections` : undefined} />
      <div className="sec-row"><span className={`dot ${card.defender === "Running" ? "good" : "bad"}`} />
        Microsoft Defender: <strong>{card.defender === "Running" ? "on" : card.defender}</strong></div>
      {card.findings?.length > 0 && (
        <ul className="findings">
          {card.findings.map((f: Card, i: number) => (
            <li key={i} className={f.severity}>
              <span className="sev">{f.severity === "high" ? "Suspicious" : "Check"}</span>
              <span><strong>{f.where}: {f.item}</strong><small>{f.reason}</small>{f.detail && <code>{f.detail}</code>}</span>
            </li>
          ))}
        </ul>
      )}
      <p className="note">A quick check for the usual hiding tricks, not a full antivirus scan.</p>
    </section>
  );
}

function CleanupCard({ card, onAction }: { card: Card; onAction: OnAction }) {
  if (!card.scanned) {
    return (
      <section className="rcard"><Head kind="cleanup" title="Clean-up" />
        <div className="cta"><span>I need to measure your folders once first.</span>
          <button className="primary" onClick={() => api.startScan().then(() => setTimeout(() => onAction("How's my storage?"), 1500))}>Scan now</button></div>
      </section>
    );
  }
  const items: { label: string; bytes: number; hint: string; path?: string }[] = [];
  if (card.recycle_bin?.bytes) items.push({ label: "Recycle Bin", bytes: card.recycle_bin.bytes, hint: `${card.recycle_bin.items} items` });
  card.caches.forEach((h: Card) => items.push({ label: h.name, bytes: h.bytes, hint: h.advice, path: h.path }));
  if (card.old_installers.bytes) items.push({ label: "Old installers in Downloads", bytes: card.old_installers.bytes, hint: `${card.old_installers.count} files older than ${card.old_installers.days} days` });
  if (card.duplicates.savings) items.push({ label: "Duplicate copies", bytes: card.duplicates.savings, hint: `${card.duplicates.count} sets of identical files` });
  items.sort((a, b) => b.bytes - a.bytes);
  const max = Math.max(1, ...items.map((i) => i.bytes));
  return (
    <section className="rcard">
      <Head kind="cleanup" title={`About ${bytes(card.total)} you could free`} sub="Safe to clear: caches, leftovers and copies, not your own files" />
      <ul className="rows">
        {items.map((i) => (
          <li key={i.label}><span className="row-name">{i.label}<small>{i.hint}</small></span>
            <Bar value={i.bytes} max={max} tone="good" /><span className="size">{bytes(i.bytes)}</span>{i.path && <Show path={i.path} />}</li>
        ))}
      </ul>
      {card.app_managed?.length > 0 && (
        <>
          <h4>Big, but managed by their app</h4>
          <ul className="rows">
            {card.app_managed.map((h: Card) => (
              <li key={h.id}><span className="row-name">{h.name}<small>{h.advice}</small></span><span className="size">{bytes(h.bytes)}</span></li>
            ))}
          </ul>
        </>
      )}
      <p className="note">Athena can't delete anything yet. Use Show to open these in File Explorer.</p>
    </section>
  );
}

function HelpCard({ card, onAction }: { card: Card; onAction: OnAction }) {
  return (
    <section className="rcard">
      <Head kind="help" title="Try asking" />
      <div className="chips-row">
        {card.examples.map((e: string) => <button key={e} className="ask-chip" onClick={() => onAction(e)}>{e}</button>)}
      </div>
    </section>
  );
}
