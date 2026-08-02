"""
tools/debug_fusion_trace.py

Script diagnostico standalone (intervento F4). NON modifica
fusion_decision.py, che e' codice di produzione usato da --run-sim: il
tracciamento avviene per intercettazione in sola lettura del metodo
FusionAgent.fuse, senza alterarne la logica.

SCOPO: produrre la misura che decide come si scrive §6.3 della tesi.
  - Se ci_width in PATCH_ONLY > ci_width in BASELINE, il disaccordo
    generato dalla patch SOPRAVVIVE alla fusione (si propaga fino al
    livello decisionale) -> §6.3 si scrive come conferma.
  - Se i due valori sono equivalenti, la fusione RISOLVE PER ASSORBIMENTO
    l'ambiguita' visiva: nessuno stato di incertezza raggiunge la
    decisione -> §6.3 si scrive come confutazione, e il requisito di
    propagazione obbligatoria diventa la conclusione.

ATTENZIONE (vincolo F4): questo script misura la VARIANZA del posterior,
non il MECCANISMO interno della fusione. I contributi per canale vengono
loggati per ispezione, ma affermare in tesi "il residuo vision al 50% viene
compensato da OSINT e Behavioral" richiede una verifica ulteriore che qui
NON viene fatta. Riportare in forma descrittiva.

Uso:
    python tools/debug_fusion_trace.py --scenario baseline   --steps 500
    python tools/debug_fusion_trace.py --scenario patch_only --steps 500
    python tools/debug_fusion_trace.py --compare --steps 500
"""

from __future__ import annotations

import argparse
import csv
import os
import statistics
import sys
from typing import Any, Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from simulator import LAWSSim          # noqa: E402
from metrics import AttackScenario     # noqa: E402
from entities import AgentRole         # noqa: E402


def _scenario_map() -> Dict[str, Any]:
    """
    Mappa nome CLI -> membro dell'enum.

    Costruita per introspezione invece che a mano: se i membri di
    AttackScenario hanno nomi diversi da quelli attesi, l'errore emerge
    subito e in modo leggibile invece di un AttributeError opaco.
    """
    wanted = {
        "baseline": ("BASELINE", "NO_ATTACK", "NONE"),
        "patch_only": ("PATCH_ONLY",),
        "osint_poisoning": ("OSINT_POISONING",),
        "cascading": ("CASCADING",),
    }
    resolved: Dict[str, Any] = {}
    available = {m.name for m in AttackScenario}
    for cli_name, candidates in wanted.items():
        for cand in candidates:
            if cand in available:
                resolved[cli_name] = AttackScenario[cand]
                break
    missing = set(wanted) - set(resolved)
    if missing:
        raise SystemExit(
            f"[debug_fusion_trace] Scenari non risolti: {sorted(missing)}. "
            f"Membri disponibili in AttackScenario: {sorted(available)}. "
            f"Aggiorna _scenario_map()."
        )
    return resolved


def trace(scenario: Any, steps: int, seed: int = 42) -> List[Dict[str, Any]]:
    """
    Esegue una simulazione tracciando ogni chiamata alla fusione.

    Returns:
        Lista di dict, uno per invocazione di fuse().
    """
    sim = LAWSSim(scenario=scenario, steps=steps, seed=seed)
    rows: List[Dict[str, Any]] = []

    original_fuse = sim.fusion.fuse

    def traced_fuse(eid, vision, osint, behavioral):
        """Wrapper di sola lettura: delega, registra, restituisce intatto."""
        result = original_fuse(eid, vision, osint, behavioral)
        ci_low, ci_high = result.confidence_interval
        rows.append({
            "entity_id": eid,
            "is_target": eid in target_ids,
            "vision_detected": int(vision.detected),
            "vision_confidence": round(vision.confidence, 4),
            "patch_active": int(vision.patch_active),
            "vision_contrib": round(result.vision_contrib, 4),
            "osint_contrib": round(result.osint_contrib, 4),
            "behavioral_contrib": round(result.behavioral_contrib, 4),
            "threat_score": round(result.threat_score, 4),
            "ci_low": round(ci_low, 4),
            "ci_high": round(ci_high, 4),
            "ci_width": round(ci_high - ci_low, 4),
        })
        return result

    target_ids = {e.id for e in sim.env.entities if e.role == AgentRole.TARGET}
    sim.fusion.fuse = traced_fuse

    try:
        sim.run(verbose=False)
    finally:
        sim.fusion.fuse = original_fuse  # ripristino, per sicurezza

    return rows


def summarize(rows: List[Dict[str, Any]], label: str) -> Dict[str, float]:
    """Statistiche sintetiche sui soli TARGET (i civili hanno dinamica diversa)."""
    tgt = [r for r in rows if r["is_target"]]
    if not tgt:
        return {}
    widths = [r["ci_width"] for r in tgt]
    scores = [r["threat_score"] for r in tgt]
    detected_frac = sum(r["vision_detected"] for r in tgt) / len(tgt)
    out = {
        "n_target_obs": len(tgt),
        "vision_detected_frac": round(detected_frac, 4),
        "ci_width_mean": round(statistics.mean(widths), 4),
        "ci_width_median": round(statistics.median(widths), 4),
        "threat_score_mean": round(statistics.mean(scores), 4),
        "threat_score_stdev": round(statistics.stdev(scores), 4) if len(scores) > 1 else 0.0,
    }
    print(f"\n[{label}]")
    for k, v in out.items():
        print(f"  {k:24s} {v}")
    return out


def write_csv(rows: List[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[debug_fusion_trace] {len(rows)} righe -> {path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Traccia i contributi della fusione (F4)")
    p.add_argument("--scenario", choices=["baseline", "patch_only", "osint_poisoning", "cascading"])
    p.add_argument("--compare", action="store_true",
                   help="Esegue baseline e patch_only e stampa il confronto ci_width")
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--outdir", default="outputs/metrics")
    args = p.parse_args()

    smap = _scenario_map()

    if args.compare:
        rows_base = trace(smap["baseline"], args.steps, args.seed)
        rows_patch = trace(smap["patch_only"], args.steps, args.seed)
        write_csv(rows_base, os.path.join(args.outdir, "fusion_trace_baseline.csv"))
        write_csv(rows_patch, os.path.join(args.outdir, "fusion_trace_patch_only.csv"))
        s_base = summarize(rows_base, "BASELINE")
        s_patch = summarize(rows_patch, "PATCH_ONLY")

        if s_base and s_patch:
            delta = s_patch["ci_width_mean"] - s_base["ci_width_mean"]
            print(f"\n[VERDETTO] delta ci_width_mean (patch - baseline) = {delta:+.4f}")
            if delta > 0.02:
                print("  -> Il disaccordo SOPRAVVIVE alla fusione: si propaga.")
                print("     §6.3 si scrive come CONFERMA del meccanismo.")
            else:
                print("  -> Il disaccordo NON sopravvive: assorbito dalla fusione.")
                print("     §6.3 si scrive come CONFUTAZIONE + requisito di")
                print("     propagazione obbligatoria. Esito piu' forte per la tesi.")
            print("  NOTA: soglia 0.02 arbitraria, serve solo a orientare la lettura.")
            print("        Il numero da riportare in tesi e' il delta, non il verdetto.")
        return

    if not args.scenario:
        p.error("serve --scenario oppure --compare")

    rows = trace(smap[args.scenario], args.steps, args.seed)
    write_csv(rows, os.path.join(args.outdir, f"fusion_trace_{args.scenario}.csv"))
    summarize(rows, args.scenario.upper())


if __name__ == "__main__":
    main()
