# Material-Management
Inventory Optimization &amp; Stockout Analysis : Analyzing ₹1,280 crore worth of inventory, identify stockout risks, assess supplier lead-time variability, and optimize reorder points to improve inventory management and supply-chain efficiency.
"""
RINL Visakhapatnam Steel Plant - Inventory & Stockout Analysis
================================================================
Reproduces the analysis in Chapters 5-6 of the project report
"Material Management at RINL Visakhapatnam Steel Plant".

Materials : Coking Coal (CC), Limestone (LS), Iron Ore Fines (IOF)
Data      : 10 fortnightly stock snapshots, 01-Nov-2024 to 16-Mar-2025 (Annexure A)

Outputs   : console report, CSV tables and PNG charts in ./output/

Requirements:  pip install pandas numpy matplotlib
Run         :  python rinl_inventory_stockout_analysis.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # works without a display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path("output")
OUT.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# 1. DATA  (Annexure A - stock in tonnes, "days" as reported by the plant)
# ---------------------------------------------------------------------------
snapshots = pd.DataFrame(
    {
        "date": pd.to_datetime(
            ["2024-11-01", "2024-11-15", "2024-12-01", "2024-12-15", "2025-01-01",
             "2025-01-15", "2025-02-02", "2025-02-16", "2025-03-02", "2025-03-16"]
        ),
        "CC_stock":  [306318, 222161, 397687, 450351, 396930, 360901, 403910, 366111, 398798, 373006],
        "CC_days":   [25, 25, 32, 36, 30, 27, 30, 27, 29, 27],
        "LS_stock":  [27944, 1000, 4000, 2164, 4946, 3224, 10325, 9271, 10370, 500],
        "LS_days":   [8, 1, 45, 1, 1, 40, 1, 1, 31, 30],
        "IOF_stock": [304014, 427213, 536945, 633412, 682393, 634583, 598612, 542764, 493986, 467192],
        "IOF_days":  [22, 31, 30, 35, 33, 30, 29, 26, 24, 23],
    }
)

# ---------------------------------------------------------------------------
# 2. PARAMETERS  (Chapters 2.5, 5.3, 5.5-5.7, 6.2)
# ---------------------------------------------------------------------------
PARAMS = {
    "Coking Coal": dict(
        code="CC", daily_use=9321, norm_days=45,
        lt_avg=45, lt_max=60,           # days (range 7-60 across origins)
        cost_per_t=25000,               # Rs / tonne
        holding_pct=0.1454,
    ),
    "Limestone": dict(
        code="LS", daily_use=3645, norm_days=15,
        lt_avg=27, lt_max=35,           # UAE single source
        cost_per_t=2500,
        holding_pct=0.1294,
    ),
    "Iron Ore Fines": dict(
        code="IOF", daily_use=16000, norm_days=20,
        lt_avg=2, lt_max=5,             # domestic rail (NMDC / KIOCL)
        cost_per_t=6757,
        holding_pct=0.1314,
    ),
}

CR = 1e7  # 1 Crore = 10,000,000 rupees

# A stock level is flagged as a "near-stockout" when it covers less than this
# many days of consumption (computed from stock / daily use).
NEAR_STOCKOUT_DAYS = 2


# ---------------------------------------------------------------------------
# 3. CORE KPIs  (Sections 5.3 - 5.8)
# ---------------------------------------------------------------------------
def build_kpis(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, p in PARAMS.items():
        s = df[f"{p['code']}_stock"]
        reported_days = df[f"{p['code']}_days"]

        avg_stock = s.mean()
        variability = (s.max() - s.min()) / avg_stock * 100          # (Max-Min)/Avg

        safety_days = (p["lt_max"] - p["lt_avg"])                    # (MaxLT - AvgLT) days
        safety_stock = safety_days * p["daily_use"]                  # tonnes
        rop = p["lt_avg"] * p["daily_use"] + safety_stock            # tonnes

        inv_value_cr = avg_stock * p["cost_per_t"] / CR
        holding_cr = inv_value_cr * p["holding_pct"]

        rows.append(
            {
                "Material": name,
                "Min Stock (T)": s.min(),
                "Max Stock (T)": s.max(),
                "Avg Stock (T)": round(avg_stock),
                "Std Dev (T)": round(s.std(ddof=1)),
                "Variability %": round(variability, 1),
                "Avg Days (reported)": round(reported_days.mean(), 1),
                "Norm Days": p["norm_days"],
                "Safety Stock (T)": safety_stock,
                "Safety Stock (days)": safety_days,
                "ROP (T)": rop,
                "ROP (days)": rop / p["daily_use"],
                "Avg Stock vs ROP (T)": round(avg_stock - rop),
                "Inv Value (Rs Cr)": round(inv_value_cr, 1),
                "Annual Holding Cost (Rs Cr)": round(holding_cr, 2),
                "Turnover (x)": round(365 / (avg_stock / p["daily_use"]), 2),
            }
        )
    k = pd.DataFrame(rows).set_index("Material")

    # ABC classification by share of total inventory value
    k = k.sort_values("Inv Value (Rs Cr)", ascending=False)
    share = k["Inv Value (Rs Cr)"] / k["Inv Value (Rs Cr)"].sum() * 100
    k["Value Share %"] = share.round(1)
    k["Cumulative %"] = share.cumsum().round(1)
    # With only three materials the report assigns A / B / C in rank order
    k["ABC"] = ["A", "B", "C"][: len(k)]
    return k


# ---------------------------------------------------------------------------
# 4. STOCKOUT / BREACH ANALYSIS  (Sections 5.3, 5.5, 5.6, 10.4)
# ---------------------------------------------------------------------------
def stockout_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Per-snapshot, per-material risk flags."""
    records = []
    for name, p in PARAMS.items():
        c = p["code"]
        safety_stock = (p["lt_max"] - p["lt_avg"]) * p["daily_use"]
        rop = p["lt_avg"] * p["daily_use"] + safety_stock
        norm_stock = p["norm_days"] * p["daily_use"]

        for _, r in df.iterrows():
            stock = r[f"{c}_stock"]
            computed_days = stock / p["daily_use"]
            records.append(
                {
                    "Date": r["date"].date(),
                    "Material": name,
                    "Stock (T)": stock,
                    "Days (reported)": r[f"{c}_days"],
                    "Days (computed)": round(computed_days, 1),
                    "Below Norm Days": r[f"{c}_days"] < p["norm_days"],
                    "Below Safety Stock": stock < safety_stock,
                    "Below ROP": stock < rop,
                    "Near-Stockout": computed_days < NEAR_STOCKOUT_DAYS,
                    "Shortfall vs Norm (T)": max(norm_stock - stock, 0),
                    "Shortfall vs Norm (Rs Cr)": round(
                        max(norm_stock - stock, 0) * p["cost_per_t"] / CR, 1),
                }
            )
    return pd.DataFrame(records)


def stockout_summary(flags: pd.DataFrame) -> pd.DataFrame:
    g = flags.groupby("Material")
    n = g.size()
    s = pd.DataFrame(
        {
            "Snapshots": n,
            "Below Norm Days": g["Below Norm Days"].sum(),
            "Below Safety Stock": g["Below Safety Stock"].sum(),
            "Below ROP": g["Below ROP"].sum(),
            "Near-Stockout Events": g["Near-Stockout"].sum(),
        }
    )
    s["% Time Below Norm"] = (s["Below Norm Days"] / s["Snapshots"] * 100).round(0)
    s["% Time Below ROP"] = (s["Below ROP"] / s["Snapshots"] * 100).round(0)
    return s


def reported_vs_computed_days(flags: pd.DataFrame) -> pd.DataFrame:
    """Data-quality check: flags snapshots where the plant's 'days' figure
    differs a lot from stock / daily consumption (notably for Limestone)."""
    f = flags.copy()
    f["Gap (days)"] = (f["Days (reported)"] - f["Days (computed)"]).round(1)
    return f[f["Gap (days)"].abs() > 10][
        ["Date", "Material", "Stock (T)", "Days (reported)", "Days (computed)", "Gap (days)"]
    ]


# ---------------------------------------------------------------------------
# 5. COKING-COAL SHORTFALL & WORKING CAPITAL  (Sections 5.3, 5.6, 6.6, 6.7)
# ---------------------------------------------------------------------------
def coking_coal_shortfalls(kpis: pd.DataFrame) -> dict:
    p = PARAMS["Coking Coal"]
    worst_day = 25                                              # Nov-2024 low
    shortfall_45d = (p["norm_days"] - worst_day) * p["daily_use"] * p["cost_per_t"] / CR
    rop_gap_t = -kpis.loc["Coking Coal", "Avg Stock vs ROP (T)"]
    rop_gap_cr = rop_gap_t * p["cost_per_t"] / CR
    avg_days = kpis.loc["Coking Coal", "Avg Days (reported)"]
    cover_gap_cr = (p["norm_days"] - avg_days) * p["daily_use"] * p["cost_per_t"] / CR
    return {
        "Worst-case shortfall vs 45-day norm (Rs Cr)": round(shortfall_45d, 0),
        "Avg stock below ROP (tonnes)": rop_gap_t,
        "Avg stock below ROP (Rs Cr)": round(rop_gap_cr, 0),
        "Extra WC to reach 45 days from avg cover (Rs Cr)": round(cover_gap_cr, 0),
    }


def working_capital_model(kpis: pd.DataFrame) -> pd.DataFrame:
    """Target = ROP-based; release if avg stock > target, invest if below."""
    rows = []
    for name, p in PARAMS.items():
        avg = kpis.loc[name, "Avg Stock (T)"]
        rop = kpis.loc[name, "ROP (T)"]
        gap = avg - rop
        rows.append(
            {
                "Material": name,
                "Avg Stock (T)": avg,
                "ROP (T)": rop,
                "Gap vs ROP (T)": gap,
                "WC Release (Rs Cr)": round(max(gap, 0) * p["cost_per_t"] / CR, 1),
                "WC Invest (Rs Cr)": round(max(-gap, 0) * p["cost_per_t"] / CR, 1),
            }
        )
    return pd.DataFrame(rows).set_index("Material")


# ---------------------------------------------------------------------------
# 6. CHARTS
# ---------------------------------------------------------------------------
def make_charts(df: pd.DataFrame, kpis: pd.DataFrame) -> None:
    colors = {"Coking Coal": "#c0392b", "Limestone": "#e67e22", "Iron Ore Fines": "#2980b9"}
    x = df["date"]

    # (a) Computed inventory days vs normative minimum, per material
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    for ax, (name, p) in zip(axes, PARAMS.items()):
        days = df[f"{p['code']}_stock"] / p["daily_use"]
        ax.plot(x, days, marker="o", color=colors[name], label="Stock cover (days)")
        ax.axhline(p["norm_days"], ls="--", color="grey", label=f"Norm = {p['norm_days']} d")
        rop_days = kpis.loc[name, "ROP (days)"]
        ax.axhline(rop_days, ls=":", color="black", label=f"ROP = {rop_days:.0f} d")
        ax.set_ylabel("Days")
        ax.set_title(name)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Stock cover vs normative minimum and reorder point", fontsize=13)
    fig.tight_layout()
    fig.savefig(OUT / "inventory_days_vs_norm.png", dpi=150)
    plt.close(fig)

    # (b) Limestone stock vs safety stock (near-stockout view)
    p = PARAMS["Limestone"]
    ss = (p["lt_max"] - p["lt_avg"]) * p["daily_use"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.bar(x, df["LS_stock"], width=8, color=colors["Limestone"], label="Stock (T)")
    ax.axhline(ss, color="red", ls="--", label=f"Safety stock = {ss:,.0f} T")
    ax.set_title("Limestone stock vs safety stock (stockout risk)")
    ax.set_ylabel("Tonnes")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "limestone_stockout_risk.png", dpi=150)
    plt.close(fig)

    # (c) Variability comparison
    fig, ax = plt.subplots(figsize=(7, 4.5))
    v = kpis["Variability %"]
    ax.bar(v.index, v.values, color=[colors[m] for m in v.index])
    for i, val in enumerate(v.values):
        ax.text(i, val + 5, f"{val:.1f}%", ha="center")
    ax.set_title("Inventory variability = (Max - Min) / Avg")
    ax.set_ylabel("%")
    fig.tight_layout()
    fig.savefig(OUT / "variability_comparison.png", dpi=150)
    plt.close(fig)

    # (d) Avg stock vs safety stock vs ROP
    fig, ax = plt.subplots(figsize=(9, 4.5))
    idx = np.arange(len(kpis))
    w = 0.27
    ax.bar(idx - w, kpis["Avg Stock (T)"], w, label="Avg stock")
    ax.bar(idx, kpis["Safety Stock (T)"], w, label="Safety stock")
    ax.bar(idx + w, kpis["ROP (T)"], w, label="Reorder point")
    ax.set_xticks(idx)
    ax.set_xticklabels(kpis.index)
    ax.set_ylabel("Tonnes")
    ax.set_title("Average stock vs safety stock vs reorder point")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "stock_vs_safety_vs_rop.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 7. MAIN
# ---------------------------------------------------------------------------
def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    kpis = build_kpis(snapshots)
    flags = stockout_analysis(snapshots)
    summary = stockout_summary(flags)
    mismatch = reported_vs_computed_days(flags)
    wc = working_capital_model(kpis)
    cc = coking_coal_shortfalls(kpis)

    print("\n=== 1. KPI SCORECARD (Ch. 5-6) ===")
    print(kpis.T.to_string())

    print("\n=== 2. STOCKOUT / BREACH SUMMARY ===")
    print(summary.to_string())

    print("\n=== 3. NEAR-STOCKOUT EVENTS (stock cover < "
          f"{NEAR_STOCKOUT_DAYS} days) ===")
    ne = flags[flags["Near-Stockout"]][
        ["Date", "Material", "Stock (T)", "Days (reported)", "Days (computed)"]]
    print(ne.to_string(index=False) if len(ne) else "None")

    print("\n=== 4. COKING COAL SHORTFALL ===")
    for k, v in cc.items():
        print(f"  {k}: {v:,.0f}")

    print("\n=== 5. WORKING-CAPITAL MODEL (ROP-based) ===")
    print(wc.to_string())

    print("\n=== 6. TOTALS ===")
    print(f"  Total avg inventory value : Rs {kpis['Inv Value (Rs Cr)'].sum():,.0f} Cr")
    print(f"  Total annual holding cost : Rs {kpis['Annual Holding Cost (Rs Cr)'].sum():,.0f} Cr")

    print("\n=== 7. DATA-QUALITY CHECK: reported vs computed days (gap > 10) ===")
    print(mismatch.to_string(index=False) if len(mismatch) else "None")

    # Save tables
    kpis.to_csv(OUT / "kpi_scorecard.csv")
    flags.to_csv(OUT / "stockout_flags_by_snapshot.csv", index=False)
    summary.to_csv(OUT / "stockout_summary.csv")
    wc.to_csv(OUT / "working_capital_model.csv")
    make_charts(snapshots, kpis)
    print(f"\nSaved CSV tables and PNG charts to: {OUT.resolve()}")


if __name__ == "__main__":
    main()
