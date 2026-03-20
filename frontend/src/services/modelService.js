const API = "/api/model";

export async function initializeModel(params) {
  const cleaned = Object.fromEntries(
    Object.entries(params).filter(([_, v]) => v !== undefined)
  );
  const res = await fetch(`${API}/initialize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(cleaned),   // was: body: params
  });
  if (!res.ok) throw await res.json();
  return res.json();
}

export async function runModel() {
  const res = await fetch(`${API}/run`, { method: "POST" });
  if (!res.ok) throw await res.json();
  return res.json();
}