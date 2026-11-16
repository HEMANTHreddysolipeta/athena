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

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
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
