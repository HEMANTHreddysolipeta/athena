import { useEffect, useState } from "react";
import type { ReactNode } from "react";
import { api, UNAUTHORIZED } from "./api";

/**
 * Signs in with the token from the launch link (http://127.0.0.1:8000/#token=...)
 * and shows a sign-in screen when there is no valid session. The token is
 * removed from the address bar (and the history entry) right away.
 */
export default function Gate({ children }: { children: ReactNode }) {
  const [state, setState] = useState<"checking" | "in" | "out">("checking");
  const [token, setToken] = useState("");
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const fromLink = new URLSearchParams(window.location.hash.slice(1)).get("token");
    if (fromLink) history.replaceState(null, "", window.location.pathname + window.location.search);
    const check = fromLink
      ? api.signIn(fromLink).catch(() => api.session())
      : api.session();
    check.then((s) => setState(s.authenticated ? "in" : "out")).catch(() => setState("out"));

    const signedOut = () => setState("out");
    window.addEventListener(UNAUTHORIZED, signedOut);
    return () => window.removeEventListener(UNAUTHORIZED, signedOut);
  }, []);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    try {
      await api.signIn(token.trim());
      setState("in");
    } catch {
      setError("That token doesn't match this Athena install.");
    }
  }

  if (state === "checking") return <div className="gate"><p className="muted">Starting Athena…</p></div>;
  if (state === "in") return <>{children}</>;
  return (
    <div className="gate">
      <form className="gate-card" onSubmit={submit}>
        <img src="/favicon.svg" alt="" width={36} height={36} />
        <h1>Sign in to Athena</h1>
        <p className="muted">
          Athena can see your files, so only you can open it. Start it with
          <code>python -m athena.serve</code>, which opens this page already signed in.
        </p>
        <p className="muted small">
          Or paste the token from the <code>token</code> file in Athena's data folder
          (Windows: <code>%LOCALAPPDATA%\Athena</code>).
        </p>
        <input type="password" value={token} onChange={(e) => setToken(e.target.value)}
          placeholder="Token" autoFocus aria-label="Token" />
        {error && <p className="error">{error}</p>}
        <button className="primary" disabled={!token.trim()}>Sign in</button>
      </form>
    </div>
  );
}
