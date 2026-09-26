"""CLI 08: budget-matched routing, sequential oracle, and Q2 (plan §12-§14 / M5).

For each high-resolution invocation rate rho (20/40/60%) we escalate the top-rho samples
chosen by each scalar uncertainty score AND by the budget-matched sequential oracle, then
compare the best scalar router to the oracle. The oracle-scalar gap answers Q2: whether a
LEARNED value router has headroom that plain uncertainty routing cannot reach.

Run it yourself (fast; per-rho progress bar):
  conda activate deepminer
  python scripts/08_routing_oracle.py --config configs/analysis.yaml
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from tqdm import tqdm

from mvwi import config as C
from mvwi.audit import oracle as OR
from mvwi.audit import outcomes as OC
from mvwi.audit import q2 as Q2
from mvwi.audit import routing as RT
from mvwi.audit import uncertainty as UN
from mvwi.preprocess.version import compute_preproc_hash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="analysis.yaml")
    args = ap.parse_args()

    cfg = C.load_config(args.config)
    logger = C.setup_logger("08_routing_oracle", cfg)
    C.set_seed(cfg["project"]["seed"])

    preproc = C.load_config("preprocessing.yaml")["preprocessing"]
    phash = compute_preproc_hash(preproc)
    tables = C.R(cfg["outputs"]["tables_dir"])
    tables.mkdir(parents=True, exist_ok=True)

    df = OC.read_outcomes(cfg, phash, "val")
    low, high = cfg["main_transition"]
    pair = UN.build_pair_frame(df, low, high)
    scores = UN.escalation_scores(pair)
    cl = pair["correct_low"].to_numpy()
    ch = pair["correct_high"].to_numpy()
    cost = cfg["cost"]
    ucfg = cfg.get("utility", {"type": "error_to_correct", "weight": 1.0})
    rhos = [float(r) for r in cfg["escalation_rates"]]
    score_names = cfg["uncertainty_scores"]

    scalar_by_rho, routing_rows = {}, []
    oracle_by_rho, oracle_rows = {}, []

    for rho in tqdm(rhos, desc="operating points", unit="rho"):
        scalar_by_rho[rho] = {}
        for name in score_names:
            m = RT.top_rho_mask(scores[name], rho)
            res = RT.routing_metrics(cl, ch, m, cost)
            res.update({"target_rho": rho, "policy": name})
            scalar_by_rho[rho][name] = res
            routing_rows.append(res)
        o = OR.sequential_oracle(cl, ch, rho, cost, ucfg)
        o.update({"target_rho": rho})
        oracle_by_rho[rho] = o
        oracle_rows.append(o)

    # clairvoyant (unbudgeted) bound -- reported in its OWN file, never at a matched rho
    clair = OR.clairvoyant_bound(cl, ch, cost)

    pd.DataFrame(routing_rows).to_csv(tables / "routing_curves.csv", index=False)
    pd.DataFrame(oracle_rows).to_csv(tables / "oracle_curves.csv", index=False)
    pd.DataFrame([clair]).to_csv(tables / "clairvoyant_bound.csv", index=False)

    for rho in rhos:
        best_name, best = Q2.pick_best_scalar(scalar_by_rho[rho])
        o = oracle_by_rho[rho]
        logger.info("rho=%.2f best_scalar=%s acc=%.4f | oracle acc=%.4f | WCrec %.3f->%.3f",
                    rho, best_name, best["final_accuracy"], o["final_accuracy"],
                    best["wc_recall"], o["wc_recall"])
    logger.info("clairvoyant unbudgeted bound: acc=%.4f at invocation_rate=%.4f",
                clair["final_accuracy"], clair["invocation_rate"])

    q2res = Q2.evaluate_q2(scalar_by_rho, oracle_by_rho, cfg["q2"])
    for p in q2res["operating_points"]:
        logger.info("Q2 point rho=%.2f acc_gap_pp=%s recall_gap_pp=%s (best_scalar=%s)",
                    p["rho"], p["acc_gap_pp"], p["recall_gap_pp"], p["best_scalar"])
    logger.info("Q2 verdict=%s", q2res["verdict"])

    C.update_metrics(cfg, "routing", {"preproc_hash": phash, "main_transition": [low, high],
                                       "cost": cost, "by_rho": {str(r): scalar_by_rho[r] for r in rhos}})
    C.update_metrics(cfg, "oracle", {"utility": ucfg,
                                      "sequential_by_rho": {str(r): oracle_by_rho[r] for r in rhos},
                                      "clairvoyant_unbudgeted": clair})
    C.update_metrics(cfg, "q2", q2res)
    C.write_run_config(cfg, script="08_routing_oracle.py",
                        extra={"preproc_hash": phash, "q2_verdict": q2res["verdict"]})
    logger.info("routing / oracle / Q2 complete")


if __name__ == "__main__":
    main()
