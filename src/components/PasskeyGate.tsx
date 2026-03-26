import { useState } from "react";
import { Lock } from "lucide-react";
import { validatePasskey, setAuthenticated } from "@/lib/passkey";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface PasskeyGateProps {
  children: React.ReactNode;
}

export function PasskeyGate({ children }: PasskeyGateProps) {
  const [authed, setAuthed] = useState(() => sessionStorage.getItem("_authenticated") === "1");
  const [token, setToken] = useState("");
  const [error, setError] = useState("");

  if (authed) return <>{children}</>;

  function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    if (!token.trim()) { setError("Please enter a passkey."); return; }
    const { valid, message } = validatePasskey(token.trim());
    if (valid) {
      setAuthenticated();
      setAuthed(true);
    } else {
      setError(message);
    }
  }

  return (
    <div className="min-h-screen bg-background flex items-center justify-center p-4">
      <div className="w-full max-w-sm space-y-6">
        <div className="flex flex-col items-center gap-3 text-center">
          <div className="h-14 w-14 rounded-2xl bg-purple-800 flex items-center justify-center shadow-lg">
            <Lock className="h-7 w-7 text-white" />
          </div>
          <h1 className="text-2xl font-bold gradient-text">Access Required</h1>
          <p className="text-muted-foreground text-sm">
            Please enter your passkey to access this application.
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-3">
          <Input
            type="password"
            placeholder="Passkey"
            value={token}
            onChange={(e) => { setToken(e.target.value); setError(""); }}
            autoFocus
          />
          {error && <p className="text-destructive text-sm">{error}</p>}
          <Button type="submit" className="w-full">
            Submit
          </Button>
        </form>
      </div>
    </div>
  );
}
