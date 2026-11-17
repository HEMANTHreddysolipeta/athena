import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { api, streamChat } from "./api";
import type { Conversation, Doc, Done, Health, Metrics, Source } from "./api";
import Storage from "./Storage";

type Turn = {
  id: number;
  question: string;
  answer: string;
  sources: Source[];
  done: Done | null;
  rewrite: string | null;
  error: string | null;
  streaming: boolean;
  startedAt: number;
};

type Tab = "chats" | "library" | "storage" | "stats";

let nextId = 1;

export default function App() {
  const [health, setHealth] = useState<Health | null>(null);
  const [docs, setDocs] = useState<Doc[]>([]);
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [conversationId, setConversationId] = useState<string | null>(null);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [tab, setTab] = useState<Tab>("chats");
  const [selected, setSelected] = useState<string[]>([]);
  const [preview, setPreview] = useState<Source | null>(null);
  const [busy, setBusy] = useState(false);
  const abortRef = useRef<AbortController | null>(null);

  const refreshDocs = useCallback(() => api.documents().then(setDocs).catch(() => {}), []);
  const refreshConversations = useCallback(() => api.conversations().then(setConversations).catch(() => {}), []);
  const refreshHealth = useCallback(() => api.health().then(setHealth).catch(() => setHealth(null)), []);

  useEffect(() => {
    refreshHealth();
    refreshDocs();
    refreshConversations();
    const timer = setInterval(refreshHealth, 15000);
    return () => clearInterval(timer);
  }, [refreshHealth, refreshDocs, refreshConversations]);

  const update = (id: number, patch: Partial<Turn> | ((t: Turn) => Partial<Turn>)) =>
    setTurns((all) => all.map((t) => (t.id === id ? { ...t, ...(typeof patch === "function" ? patch(t) : patch) } : t)));

  async function ask(question: string) {
    const id = nextId++;
    setTurns((all) => [...all, { id, question, answer: "", sources: [], done: null, rewrite: null,
      error: null, streaming: true, startedAt: performance.now() }]);
    setBusy(true);
    const controller = new AbortController();
    abortRef.current = controller;
    try {
      await streamChat(
        { message: question, conversation_id: conversationId, files: selected.length ? selected : null },
        {
          conversation: (cid) => setConversationId(cid),
          rewrite: (q) => update(id, { rewrite: q }),
          sources: (sources) => update(id, { sources }),
          token: (text) => update(id, (t) => ({ answer: t.answer + text })),
          done: (done) => update(id, { done, answer: done.answer, streaming: false }),
          error: (detail) => update(id, { error: detail, streaming: false }),
        },
        controller.signal,
      );
    } catch (e) {
      if ((e as Error).name !== "AbortError") update(id, { error: (e as Error).message });
    } finally {
      update(id, { streaming: false });
      setBusy(false);
      refreshConversations();
    }
  }

  async function openConversation(id: string) {
    abortRef.current?.abort();
    const { messages } = await api.conversation(id);
    const loaded: Turn[] = [];
    for (let i = 0; i < messages.length; i++) {
      const m = messages[i];
      if (m.role !== "user") continue;
      const reply = messages[i + 1]?.role === "assistant" ? messages[i + 1] : null;
      loaded.push({
        id: nextId++, question: m.content, answer: reply?.content ?? "",
        sources: reply?.meta?.sources ?? [], done: reply?.meta ?? null, rewrite: reply?.meta?.standalone_question ?? null,
        error: reply ? null : "No answer was saved for this question.", streaming: false, startedAt: 0,
      });
    }
    setConversationId(id);
    setTurns(loaded);
    setTab("chats");
  }

  function newChat() {
    abortRef.current?.abort();
    setConversationId(null);
    setTurns([]);
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <img src="/favicon.svg" alt="" width={28} height={28} />
          <div>
            <strong>Athena</strong>
            <span>Private assistant for your files</span>
          </div>
        </div>
        <HealthBadge health={health} />
        <nav className="tabs">
          {(["chats", "library", "storage", "stats"] as Tab[]).map((t) => (
            <button key={t} className={tab === t ? "active" : ""} onClick={() => setTab(t)}>
              {t === "chats" ? "Chats" : t === "library" ? `Library (${docs.length})` : t === "storage" ? "Storage" : "Stats"}
            </button>
          ))}
        </nav>
        {tab === "chats" && (
          <ChatList
            conversations={conversations}
            current={conversationId}
            onNew={newChat}
            onOpen={openConversation}
            onDelete={async (id) => {
              await api.deleteConversation(id);
              if (id === conversationId) newChat();
              refreshConversations();
            }}
          />
        )}
        {tab === "library" && (
          <Library
            docs={docs}
            selected={selected}
            setSelected={setSelected}
            onChanged={() => { refreshDocs(); refreshHealth(); }}
          />
        )}
        {tab === "storage" && (
          <div className="panel">
            <p className="small">See how full your drives are, which folders and files are biggest, and what you
              could clean up.</p>
            <p className="hint">Athena only reads names and sizes in your home folder. Nothing leaves this
              computer, and nothing is changed. You can also type <code>/storage</code> in a chat.</p>
            <button onClick={() => setTab("chats")}>Back to chat</button>
          </div>
        )}
        {tab === "stats" && <Stats />}
      </aside>

      <main className="main">
        {tab === "storage" ? <Storage /> : (
          <>
            <Thread turns={turns} onCite={setPreview} onExample={ask} hasDocs={docs.length > 0} />
            <Composer
              busy={busy}
              onAsk={ask}
              onStop={() => abortRef.current?.abort()}
              selected={selected}
              clearSelected={() => setSelected([])}
            />
          </>
        )}
      </main>

      {preview && <SourcePreview source={preview} onClose={() => setPreview(null)} />}
    </div>
  );
}

function HealthBadge({ health }: { health: Health | null }) {
  if (!health) return <div className="health bad">● Backend not reachable</div>;
  return (
    <div className={`health ${health.llm.ok ? "ok" : "bad"}`} title={health.llm.detail}>
      ● {health.llm.ok ? health.llm.model : "Model offline"} · {health.index.chunks.toLocaleString()} chunks
      <small>{health.llm.ok ? `local · ${health.retrieval.mode}${health.retrieval.rerank ? " + rerank" : ""}` : health.llm.detail}</small>
    </div>
  );
}

function ChatList(props: {
  conversations: Conversation[];
  current: string | null;
  onNew: () => void;
  onOpen: (id: string) => void;
  onDelete: (id: string) => void;
}) {
  return (
    <div className="panel">
      <button className="primary wide" onClick={props.onNew}>+ New chat</button>
      <ul className="list">
        {props.conversations.map((c) => (
          <li key={c.id} className={c.id === props.current ? "current" : ""}>
            <button className="link" onClick={() => props.onOpen(c.id)} title={c.title}>{c.title}</button>
            <button className="icon" aria-label="Delete chat" onClick={() => props.onDelete(c.id)}>×</button>
          </li>
        ))}
        {!props.conversations.length && <li className="muted">No chats yet.</li>}
      </ul>
    </div>
  );
}

function Library(props: {
  docs: Doc[];
  selected: string[];
  setSelected: (s: string[]) => void;
  onChanged: () => void;
}) {
  const [status, setStatus] = useState<string | null>(null);
  const [url, setUrl] = useState("");
  const [dragging, setDragging] = useState(false);
  const fileInput = useRef<HTMLInputElement>(null);

  async function uploadFiles(files: FileList | File[]) {
    for (const file of Array.from(files)) {
      setStatus(`Indexing ${file.name}…`);
      try {
        const r = await api.upload(file);
        setStatus(r.status === "indexed"
          ? `Added ${r.filename}: ${r.chunks} chunks${r.ocr_pages.length ? `, OCR on ${r.ocr_pages.length} page(s)` : ""}`
          : `${r.filename}: ${r.status}`);
      } catch (e) {
        setStatus(`Failed: ${(e as Error).message}`);
      }
      props.onChanged();
    }
  }

  async function addUrl(e: React.FormEvent) {
    e.preventDefault();
    if (!url.trim()) return;
    setStatus(`Fetching ${url}…`);
    try {
      const r = await api.addUrl(url.trim());
      setStatus(`Added ${r.filename}: ${r.chunks} chunks`);
      setUrl("");
    } catch (err) {
      setStatus(`Failed: ${(err as Error).message}`);
    }
    props.onChanged();
  }

  async function reindex() {
    await api.reindex();
    setStatus("Scanning the data folder…");
    const poll = setInterval(async () => {
      const s = await api.reindexStatus();
      if (!s.running) {
        clearInterval(poll);
        const c = s.last?.counts ?? {};
        setStatus(`Data folder: ${c.indexed ?? 0} new, ${c.unchanged ?? 0} unchanged, ${c.failed ?? 0} failed`);
        props.onChanged();
      }
    }, 2000);
  }

  const toggle = (name: string) =>
    props.setSelected(props.selected.includes(name) ? props.selected.filter((n) => n !== name) : [...props.selected, name]);

  return (
    <div className="panel">
      <div
        className={`dropzone ${dragging ? "over" : ""}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); uploadFiles(e.dataTransfer.files); }}
        onClick={() => fileInput.current?.click()}
      >
        Drop files or click to upload
        <small>PDF · DOCX · PPTX · MD · TXT · HTML — scanned PDFs are OCR'd</small>
        <input ref={fileInput} type="file" multiple hidden
          accept=".pdf,.docx,.pptx,.md,.markdown,.txt,.html,.htm"
          onChange={(e) => e.target.files && uploadFiles(e.target.files)} />
      </div>
      <form className="url" onSubmit={addUrl}>
        <input value={url} onChange={(e) => setUrl(e.target.value)} placeholder="https://… add a web page" />
        <button type="submit">Add</button>
      </form>
      <button className="wide" onClick={reindex}>Rescan data folder</button>
      {status && <p className="status">{status}</p>}
      <p className="hint">Tick documents to ask only about them.</p>
      <ul className="list docs">
        {props.docs.map((d) => (
          <li key={d.filepath}>
            <label title={d.filename}>
              <input type="checkbox" checked={props.selected.includes(d.filename)} onChange={() => toggle(d.filename)} />
              <span className="doc-title">{d.title}</span>
              <span className="meta">
                <span className="badge">{d.filetype.replace(".", "")}</span>
                {d.pages ? `${d.pages} p · ` : ""}{d.chunks} chunks{d.ocr ? " · OCR" : ""}{d.source_url ? " · web" : ""}
              </span>
            </label>
            <button className="icon" aria-label={`Remove ${d.filename}`} onClick={async () => {
              await api.deleteDocument(d.filename);
              props.setSelected(props.selected.filter((n) => n !== d.filename));
              props.onChanged();
            }}>×</button>
          </li>
        ))}
        {!props.docs.length && <li className="muted">No documents yet. Upload some, or put files in the data folder and rescan.</li>}
      </ul>
    </div>
  );
}

function Stats() {
  const [m, setM] = useState<Metrics | null>(null);
  useEffect(() => { api.metrics().then(setM).catch(() => {}); }, []);
  if (!m) return <div className="panel muted">Loading…</div>;
  const s = (ms: number | null) => (ms == null ? "–" : `${(ms / 1000).toFixed(1)} s`);
  const rows: [string, string | number][] = [
    ["Questions asked", m.queries],
    ["Answered", m.answered],
    ["Not found", m.not_found],
    ["…decided without calling the LLM", m.abstained_without_llm],
    ["Answers fully grounded", m.grounded_answers],
    ["Answers with flagged claims", m.flagged_answers],
    ["Median time per answer", s(m.latency_ms.p50)],
    ["95th percentile", s(m.latency_ms.p95)],
    ["Tokens read / written", `${m.tokens.prompt.toLocaleString()} / ${m.tokens.output.toLocaleString()}`],
  ];
  return (
    <div className="panel">
      <table className="stats"><tbody>
        {rows.map(([k, v]) => <tr key={k}><td>{k}</td><td>{v}</td></tr>)}
      </tbody></table>
      <p className="hint">Measured on this machine. Everything runs locally; no tokens are billed.</p>
    </div>
  );
}

const EXAMPLES = [
  "What is the minimum password length when a password is the only factor?",
  "What are the four functions of the AI RMF Core?",
  "What is a Denial of Wallet attack?",
  "How can you stop an AI agent from taking a high-impact action on its own?",
];

function Thread({ turns, onCite, onExample, hasDocs }: {
  turns: Turn[];
  onCite: (s: Source) => void;
  onExample: (q: string) => void;
  hasDocs: boolean;
}) {
  const end = useRef<HTMLDivElement>(null);
  useEffect(() => { end.current?.scrollIntoView({ behavior: "smooth", block: "end" }); }, [turns]);

  if (!turns.length) {
    return (
      <div className="empty">
        <h1>Ask your documents</h1>
        <p>Answers come only from your files, with a citation for every fact. Runs entirely on this computer.</p>
        {hasDocs ? (
          <div className="examples">
            {EXAMPLES.map((q) => <button key={q} onClick={() => onExample(q)}>{q}</button>)}
          </div>
        ) : (
          <p className="muted">Start by adding documents in the Library tab.</p>
        )}
        <p className="muted small">Commands: /find, /read, /open, /remember, /recall, /help</p>
      </div>
    );
  }
  return (
    <div className="thread">
      {turns.map((t) => <TurnView key={t.id} turn={t} onCite={onCite} />)}
      <div ref={end} />
    </div>
  );
}

function TurnView({ turn, onCite }: { turn: Turn; onCite: (s: Source) => void }) {
  const done = turn.done;
  const waiting = turn.streaming && !turn.answer;
  const elapsed = useElapsed(turn.streaming ? turn.startedAt : null);
  const bySource = useMemo(() => new Map(turn.sources.map((s) => [s.number, s])), [turn.sources]);
  const cited = done?.cited?.length ? turn.sources.filter((s) => done.cited.includes(s.number)) : turn.sources;
  const notFound = done && !done.found;

  return (
    <article className="turn">
      <div className="question">{turn.question}</div>
      {turn.rewrite && <div className="rewrite">Searched for: “{turn.rewrite}”</div>}
      <div className={`answer ${notFound ? "notfound" : ""} ${done?.command ? "command" : ""}`}>
        {waiting && (
          <span className="thinking">
            {turn.sources.length ? `Reading ${turn.sources.length} passages` : "Searching your documents"}… {elapsed}s
          </span>
        )}
        {turn.answer && <AnswerText text={turn.answer} sources={bySource} onCite={onCite} />}
        {turn.streaming && turn.answer && <span className="cursor">▍</span>}
        {turn.error && <p className="error">⚠ {turn.error}</p>}
      </div>
      {done && !done.command && <Checks done={done} />}
      {!notFound && cited.length > 0 && !done?.command && (
        <div className="sources">
          {cited.map((s) => (
            <button key={s.number} className="source" onClick={() => onCite(s)}>
              <span className="num">{s.number}</span>
              <span className="src-body">
                <strong>{s.title || s.filename}</strong>
                <small>{[s.page ? `p. ${s.page}` : null, s.section].filter(Boolean).join(" · ")}</small>
              </span>
              {s.relevance != null && <span className="rel" title="Reranker relevance">{Math.round(s.relevance * 100)}%</span>}
            </button>
          ))}
        </div>
      )}
    </article>
  );
}

function Checks({ done }: { done: Done }) {
  const g = done.grounding;
  const t = done.timings;
  const items: JSX.Element[] = [];
  if (!done.found) {
    items.push(<span key="nf" className="chip neutral">
      {done.abstained_early ? "Nothing relevant found — the model wasn't asked" : "The sources don't contain this"}
    </span>);
  } else if (g) {
    items.push(g.supported
      ? <span key="g" className="chip good" title="Every number, code and acronym appears in the cited sources">
          ✓ {g.checked ? `${g.checked} specific claim${g.checked > 1 ? "s" : ""} checked against sources` : "Grounded in sources"}
        </span>
      : <span key="g" className="chip warn">
          ⚠ {g.unsupported.length ? `Not in the sources: ${g.unsupported.join(", ")} — verify before relying on it` : "Cites a source that doesn't exist"}
        </span>);
  }
  if (t) items.push(<span key="t" className="chip neutral">{(t.total_ms / 1000).toFixed(1)} s</span>);
  return <div className="checks">{items}</div>;
}

function AnswerText({ text, sources, onCite }: {
  text: string;
  sources: Map<number, Source>;
  onCite: (s: Source) => void;
}) {
  // Turn [1], [2][3] and [1, 2] into clickable chips; keep line breaks.
  const parts = text.split(/(\[\d+(?:\s*,\s*\d+)*\])/g);
  return (
    <div className="answer-text">
      {parts.map((part, i) => {
        const m = part.match(/^\[(\d+(?:\s*,\s*\d+)*)\]$/);
        if (!m) return <span key={i}>{renderInline(part)}</span>;
        return m[1].split(",").map((n) => {
          const s = sources.get(Number(n.trim()));
          return s
            ? <button key={`${i}-${n}`} className="cite" onClick={() => onCite(s)} title={`${s.filename}${s.page ? `, page ${s.page}` : ""}`}>{n.trim()}</button>
            : <span key={`${i}-${n}`} className="cite missing">{n.trim()}</span>;
        });
      })}
    </div>
  );
}

function renderInline(text: string) {
  // Minimal markdown: **bold**; everything else is plain text (no HTML injection).
  return text.split(/(\*\*[^*]+\*\*)/g).map((p, i) =>
    p.startsWith("**") && p.endsWith("**") ? <strong key={i}>{p.slice(2, -2)}</strong> : p);
}

function SourcePreview({ source, onClose }: { source: Source; onClose: () => void }) {
  const [showPage, setShowPage] = useState(false);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  const isPdf = source.filename.toLowerCase().endsWith(".pdf") && source.page;
  return (
    <div className="overlay" onClick={onClose}>
      <div className="modal" onClick={(e) => e.stopPropagation()} role="dialog" aria-label="Source">
        <header>
          <div>
            <span className="num">{source.number}</span>
            <strong>{source.title || source.filename}</strong>
            <small>{source.filename}{source.page ? ` · page ${source.page}` : ""}{source.section ? ` · ${source.section}` : ""}</small>
          </div>
          <button className="icon" onClick={onClose} aria-label="Close">×</button>
        </header>
        {isPdf && (
          <div className="toggle">
            <button className={!showPage ? "active" : ""} onClick={() => setShowPage(false)}>Passage</button>
            <button className={showPage ? "active" : ""} onClick={() => setShowPage(true)}>Original page</button>
          </div>
        )}
        {showPage && isPdf
          ? <img className="page" src={api.pageImage(source.filename, source.page!)} alt={`Page ${source.page}`} />
          : <pre className="passage">{source.text}</pre>}
      </div>
    </div>
  );
}

function Composer({ busy, onAsk, onStop, selected, clearSelected }: {
  busy: boolean;
  onAsk: (q: string) => void;
  onStop: () => void;
  selected: string[];
  clearSelected: () => void;
}) {
  const [text, setText] = useState("");
  function submit() {
    const q = text.trim();
    if (!q || busy) return;
    onAsk(q);
    setText("");
  }
  return (
    <div className="composer">
      {selected.length > 0 && (
        <div className="scope">
          Asking only: {selected.join(", ")} <button className="link" onClick={clearSelected}>clear</button>
        </div>
      )}
      <div className="box">
        <textarea
          value={text}
          rows={1}
          placeholder="Ask a question about your documents…"
          onChange={(e) => setText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); submit(); }
          }}
        />
        {busy
          ? <button className="stop" onClick={onStop}>Stop</button>
          : <button className="primary" onClick={submit} disabled={!text.trim()}>Ask</button>}
      </div>
    </div>
  );
}

function useElapsed(since: number | null) {
  const [now, setNow] = useState(performance.now());
  useEffect(() => {
    if (since == null) return;
    const timer = setInterval(() => setNow(performance.now()), 250);
    return () => clearInterval(timer);
  }, [since]);
  return since == null ? 0 : Math.max(0, Math.round((now - since) / 1000));
}
