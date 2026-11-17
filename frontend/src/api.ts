// Typed client for the Athena API (backend/app.py).

export type Source = {
  number: number;
  filename: string;
  page: number | null;
  title: string;
  section: string | null;
  relevance: number | null;
  text: string;
  chunk_id: string;
};

export type Grounding = {
  supported: boolean;
  unsupported: string[];
  checked: number;
  invalid_citations: number[];
  uncited_sentences: number;
};

export type Done = {
  answer: string;
  found: boolean;
  cited: number[];
  command?: string;
  standalone_question?: string | null;
  abstained_early?: boolean;
  grounding?: Grounding | null;
  timings?: { retrieval_ms: number; generation_ms: number; total_ms: number };
  top_score?: number | null;
  llm?: { prompt_tokens?: number; output_tokens?: number };
};

export type Doc = {
  filename: string;
  filepath: string;
  title: string;
  filetype: string;
  chunks: number;
  pages: number | null;
  ocr: boolean;
  source_url: string | null;
};

export type Conversation = { id: string; title: string; created: number; updated: number };

export type StoredMessage = {
  role: "user" | "assistant";
  content: string;
  created: number;
  meta: (Done & { sources?: Source[] }) | null;
};

export type Health = {
  status: "ok" | "degraded";
  llm: { ok: boolean; detail: string; model: string };
  index: { chunks: number; documents: number; embedding_model: string };
  retrieval: { mode: string; rerank: boolean; reranker: string; top_k: number };
};

export type Metrics = {
  queries: number;
  answered: number;
  not_found: number;
  abstained_without_llm: number;
  grounded_answers: number;
  flagged_answers: number;
  latency_ms: { p50: number | null; p95: number | null };
  generation_ms: { p50: number | null };
  tokens: { prompt: number; output: number };
};

export type Drive = { mount: string; total: number; used: number; free: number; percent: number };
export type FolderEntry = { path: string; bytes: number; files: number };
export type FileEntry = { path: string; bytes: number; modified?: number };
export type FolderNode = {
  path: string; bytes: number; files: number; own_bytes: number;
  folders: FolderEntry[]; small_folders: { count: number; bytes: number };
};
export type FolderView = FolderNode & { files: FileEntry[]; file_count: number; crumbs: { name: string; path: string }[] };
export type ScanStatus = {
  state: "idle" | "scanning" | "duplicates" | "done" | "cancelled" | "error";
  error: string | null; elapsed_s: number | null;
  files?: number; bytes?: number; current?: string | null; expected_files?: number | null; hashed_bytes?: number;
};
export type SpaceHog = { id: string; name: string; path: string; bytes: number; advice: string };
export type StorageReport = {
  root: string; scanned_at: number; duration_s: number; files: number; bytes: number; unreadable_folders: number;
  home: FolderNode; largest_files: FileEntry[]; space_hogs: SpaceHog[];
  old_installers: { days: number; bytes: number; count: number; files: FileEntry[] };
  duplicates: { groups: { bytes: number; paths: string[]; savings: number }[]; count: number; savings: number;
    complete: boolean; min_bytes: number };
};
export type StorageOverview = {
  drives: Drive[]; recycle_bin: { bytes: number; items: number } | null; scan: ScanStatus; report: StorageReport | null;
};

/** Fired on any 401 so the app can show the sign-in screen. */
export const UNAUTHORIZED = "athena:unauthorized";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (response.status === 401) window.dispatchEvent(new Event(UNAUTHORIZED));
  if (!response.ok) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

const json = (method: string, body: unknown): RequestInit => ({
  method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
});

export const api = {
  health: () => request<Health>("/api/health"),
  metrics: () => request<Metrics>("/api/metrics"),
  documents: () => request<Doc[]>("/api/documents"),
  upload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<{ filename: string; status: string; chunks: number; ocr_pages: number[] }>(
      "/api/documents", { method: "POST", body: form });
  },
  addUrl: (url: string) =>
    request<{ filename: string; status: string; chunks: number }>("/api/documents/url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    }),
  deleteDocument: (filename: string) =>
    request(`/api/documents/${encodeURIComponent(filename)}`, { method: "DELETE" }),
  reindex: () => request<{ started: boolean }>("/api/reindex", { method: "POST" }),
  reindexStatus: () =>
    request<{ running: boolean; last: { counts: Record<string, number>; seconds: number } | null }>("/api/reindex"),
  conversations: () => request<Conversation[]>("/api/conversations"),
  conversation: (id: string) => request<{ id: string; messages: StoredMessage[] }>(`/api/conversations/${id}`),
  deleteConversation: (id: string) => request(`/api/conversations/${id}`, { method: "DELETE" }),
  session: () => request<{ authenticated: boolean }>("/api/session"),
  signIn: (token: string) => request<{ authenticated: boolean }>("/api/session", json("POST", { token })),
  storage: () => request<StorageOverview>("/api/assistant/storage"),
  startScan: () => request<{ started: boolean; scan: ScanStatus }>("/api/assistant/storage/scan", { method: "POST" }),
  cancelScan: () => request<{ scan: ScanStatus }>("/api/assistant/storage/scan/cancel", { method: "POST" }),
  folder: (path: string) => request<FolderView>(`/api/assistant/storage/folder?path=${encodeURIComponent(path)}`),
  reveal: (path: string) => request<{ ok: boolean }>("/api/assistant/reveal", json("POST", { path })),
  pageImage: (filename: string, page: number) =>
    `/api/documents/${encodeURIComponent(filename)}/pages/${page}.png`,
};

export type StreamHandlers = {
  conversation: (id: string) => void;
  rewrite: (question: string) => void;
  sources: (sources: Source[]) => void;
  token: (text: string) => void;
  done: (done: Done) => void;
  error: (detail: string) => void;
};

/** POST a question and read the server-sent events as they arrive. */
export async function streamChat(
  body: { message: string; conversation_id?: string | null; files?: string[] | null },
  on: StreamHandlers,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  if (response.status === 401) window.dispatchEvent(new Event(UNAUTHORIZED));
  if (!response.ok || !response.body) {
    let detail = response.statusText;
    try {
      detail = (await response.json()).detail ?? detail;
    } catch {
      /* not JSON */
    }
    on.error(detail);
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary: number;
    while ((boundary = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      let event = "message";
      let data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event: ")) event = line.slice(7);
        else if (line.startsWith("data: ")) data += line.slice(6);
      }
      if (!data) continue;
      const payload = JSON.parse(data);
      switch (event) {
        case "conversation": on.conversation(payload.id); break;
        case "rewrite": on.rewrite(payload.question); break;
        case "sources": on.sources(payload.sources); break;
        case "token": on.token(payload.text); break;
        case "done": on.done(payload); break;
        case "error": on.error(payload.detail); break;
      }
    }
  }
}
