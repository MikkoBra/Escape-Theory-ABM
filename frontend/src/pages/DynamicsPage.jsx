import { useState } from "react";
import DynamicsChart from "../components/DynamicsChart";
import Spinner from "../components/Spinner";
import InputField from "../components/InputField";
import Stat from "../components/Stat";
import { initializeModel, runModel } from "../services/modelService";

export default function DynamicsPage() {
  const [nAgents, setNAgents] = useState("");
  const [dt, setDt]           = useState("");
  const [endTime, setEndTime] = useState("");
  const [status, setStatus]   = useState("idle");
  const [stressData, setStressData] = useState(null);
  const [aversiveData, setAversiveData] = useState(null);
  const [agentIdx, setAgentIdx]     = useState(0);
  const [error, setError]           = useState(null);

  const nAgentsResolved = stressData?.[0]?.length ?? null;
  const dtResolved      = dt ? parseFloat(dt) : 1/24/60;

  const stressSeries = stressData
    ? stressData.map(row => row[Math.min(agentIdx, row.length - 1)] ?? 0)
    : [];
  const aversiveSeries = aversiveData
    ? aversiveData.map(row => row[Math.min(agentIdx, row.length - 1)] ?? 0)
    : [];

  async function handleRun() {
    setStatus("running");
    setError(null);
    setStressData(null);
    try {
      await initializeModel({
        n_agents: nAgents ? parseInt(nAgents)  : undefined,
        dt:       dt      ? parseFloat(dt)     : undefined,
        end_time: endTime ? parseInt(endTime)  : undefined,
      });
      const res = await runModel();
      setStressData(res.stress);
      setAversiveData(res.aversive_state);
      setStatus("done");
    } catch (e) {
      setError(e?.error ?? "An error occurred.");
      setStatus("error");
    }
  }

  return (
    <div className="app">

      {/* ── Sidebar ───────────────────────────────────────────── */}
      <div className="sidebar">
        <div className="sidebar-title">
          <span>Escape Theory Agent Model</span>
          Agent Dynamics
        </div>

        <div className="sidebar-section">
          <div className="section-label">Model parameters</div>
          <InputField label="N(agents)"  value={nAgents}  onChange={setNAgents}  placeholder="15" />
          <InputField label="dt"          value={dt}        onChange={setDt}        placeholder="1/1440" />
          <InputField label="T (end time)" value={endTime} onChange={setEndTime}   placeholder="50" />
        </div>

        <button
          className={`run-btn ${status === "running" ? "running" : ""}`}
          onClick={handleRun}
          disabled={status === "running"}
        >
          {status === "running" ? "Running simulation…" : "Run simulation"}
        </button>

        {error && <div className="error-box">{error}</div>}

        {stressData && (
          <div className="sidebar-section">
            <div className="section-label">Agent selection</div>
            <InputField
              label={`Agent index (0 – ${(nAgentsResolved ?? 1) - 1})`}
              value={String(agentIdx)}
              onChange={v => {
                const n = parseInt(v);
                if (!isNaN(n)) setAgentIdx(Math.max(0, Math.min(n, (nAgentsResolved ?? 1) - 1)));
              }}
              placeholder="0"
            />
            <input
              type="range"
              min={0}
              max={(nAgentsResolved ?? 1) - 1}
              value={agentIdx}
              onChange={e => setAgentIdx(parseInt(e.target.value))}
              className="agent-slider"
            />
            <div className="stats-row">
              <Stat label="Steps"  value={stressData.length} />
              <Stat label="N"      value={nAgentsResolved} />
              <Stat label="Agent"  value={agentIdx} />
            </div>
          </div>
        )}
      </div>

      {/* ── Main ──────────────────────────────────────────────── */}
      <div className="main">
        {status !== "idle" && (
          <div className="chart-header">
            <p className="chart-title">
              {status === "done"
                ? `Fig. 1. Parameter trajectories, agent ${agentIdx} of ${nAgentsResolved} — dt = ${dtResolved.toFixed(5)}`
                : status === "running"
                ? "Running simulation…"
                : "Simulation error"}
            </p>
          </div>
        )}

        <div className="chart-area">
          {status === "idle" && (
            <div className="placeholder-text">
              <em>No simulation data</em>
              Configure parameters and run the simulation.
            </div>
          )}
          {status === "running" && <Spinner />}
          {status === "error" && (
            <div className="placeholder-text error-text">{error}</div>
          )}
          {status === "done" && stressData && (
            <DynamicsChart
              dt={dtResolved}
              series={[
                { name: "S(t)", data: stressSeries, color: "#2a5c8a" },
                { name: "A(t)", data: aversiveSeries, color: "#c94f4f" },
              ]}
            />
          )}
        </div>
      </div>

    </div>
  );
}