# 07_plot_ci_vs_tc_error_fe_stacked.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf

# ============================================================
# CONFIG
# ============================================================

FITS_PATH = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_fits.csv"
CONFIDENCE_PATH = "daily_fullscale/eurostoxx600_lppls_daily_fullscale_positive_confidence.csv"
PRICE_PATH = "eurostoxx600_prices.csv"

OUTPUT_PNG = "ci_vs_tc_error_fe_stacked.png"
OUTPUT_PANEL = "ci_vs_tc_error_fe_panel.csv"

DATE_COL = "Date"
EVENT_COL = "tc_literature"
CI_COL = "positive_bubble_confidence"

LOOKBACK_TRADING_DAYS = 250
VALID_ONLY = True
MAX_ABS_ERROR_FOR_PLOT = 750

TOP_POINT_COLOR = "red"
BOTTOM_POINT_COLOR = "#0000cc"
LINE_COLOR = "black"

# Final PNG aspect ratio = 17.76 : 11.98
FIGSIZE = (17.76, 11.98)

# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "Arial",
    "font.size": 34,
    "axes.labelsize": 34,
    "axes.titlesize": 34,
    "xtick.labelsize": 34,
    "ytick.labelsize": 34,
    "legend.fontsize": 26,
})

# ============================================================
# LOAD DATA
# ============================================================

fits = pd.read_csv(FITS_PATH)
confidence = pd.read_csv(CONFIDENCE_PATH)
prices = pd.read_csv(PRICE_PATH)

for col in ["t1", "t2", "tc_predicted"]:
    fits[col] = pd.to_datetime(fits[col], errors="coerce")

confidence["t2"] = pd.to_datetime(confidence["t2"], errors="coerce")
prices[DATE_COL] = pd.to_datetime(prices[DATE_COL], errors="coerce")

fits["positive_lppls_valid"] = (
    fits["positive_lppls_valid"]
    .astype(str)
    .str.lower()
    .eq("true")
)

prices = prices.sort_values(DATE_COL).reset_index(drop=True)

fits = fits.merge(
    confidence[["t2", CI_COL]].dropna(),
    on="t2",
    how="left"
)

events = prices.loc[
    prices[EVENT_COL] == 1,
    DATE_COL
].dropna().sort_values().to_list()

# ============================================================
# BUILD PANEL
# ============================================================

chunks = []

for event_date in events:
    event_idx = prices.index[prices[DATE_COL] == event_date][0]
    start_idx = max(0, event_idx - LOOKBACK_TRADING_DAYS)
    start_date = prices.loc[start_idx, DATE_COL]

    g = fits[
        (fits["t2"] >= start_date)
        & (fits["t2"] < event_date)
        & (fits["tc_predicted"].notna())
        & (fits[CI_COL].notna())
    ].copy()

    if VALID_ONLY:
        g = g[g["positive_lppls_valid"]].copy()

    if len(g) == 0:
        continue

    g["event_date"] = event_date.strftime("%Y-%m-%d")
    g["signed_tc_error"] = (g["tc_predicted"] - event_date).dt.days
    g["absolute_tc_error"] = g["signed_tc_error"].abs()

    chunks.append(g)

if len(chunks) == 0:
    raise ValueError("No valid event-fit observations found.")

panel = pd.concat(chunks, ignore_index=True)

panel = panel[
    [CI_COL, "signed_tc_error", "absolute_tc_error", "event_date"]
].dropna().copy()

panel = panel.rename(columns={CI_COL: "ci"})

if MAX_ABS_ERROR_FOR_PLOT is not None:
    panel = panel[panel["absolute_tc_error"] <= MAX_ABS_ERROR_FOR_PLOT].copy()

panel.to_csv(OUTPUT_PANEL, index=False)

# ============================================================
# EVENT FE REGRESSIONS WITH HC3 ROBUST SE
# ============================================================

signed_fe = smf.ols(
    "signed_tc_error ~ ci + C(event_date)",
    data=panel
).fit(cov_type="HC3")

abs_fe = smf.ols(
    "absolute_tc_error ~ ci + C(event_date)",
    data=panel
).fit(cov_type="HC3")


def p_format(p):
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


print("\nEvent FE, HC3 results:")
print(
    f"Signed error: beta={signed_fe.params['ci']:.2f}, "
    f"p={p_format(signed_fe.pvalues['ci'])}, "
    f"R2={signed_fe.rsquared:.4f}"
)
print(
    f"Absolute error: beta={abs_fe.params['ci']:.2f}, "
    f"p={p_format(abs_fe.pvalues['ci'])}, "
    f"R2={abs_fe.rsquared:.4f}"
)

# ============================================================
# PLOT HELPERS
# ============================================================

def add_fe_lines(ax, model):
    ci_grid = np.linspace(panel["ci"].min(), panel["ci"].max(), 100)

    for event_date in sorted(panel["event_date"].unique()):
        pred_df = pd.DataFrame({
            "ci": ci_grid,
            "event_date": event_date
        })

        y_hat = model.predict(pred_df)

        ax.plot(
            ci_grid,
            y_hat,
            color=LINE_COLOR,
            linewidth=2.2,
            alpha=0.55
        )


def make_panel(ax, y_col, y_label, model, point_color):
    ax.scatter(
        panel["ci"],
        panel[y_col],
        s=28,
        alpha=0.25,
        color=point_color,
        edgecolor="none"
    )

    add_fe_lines(ax, model)

    beta = model.params["ci"]

    ax.set_ylabel(y_label)

    # Put beta in top-right corner like an annotation/legend
    ax.text(
        0.98,
        0.98,
        rf"$\beta$ = {beta:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=30,
        bbox=dict(facecolor="white", edgecolor="none", alpha=0.8, pad=4)
    )

    ax.grid(True, which="major", alpha=0.30, linewidth=1.2)
    ax.minorticks_off()


# ============================================================
# MAKE STACKED PLOT
# ============================================================

fig, axes = plt.subplots(
    2,
    1,
    figsize=FIGSIZE,
    sharex=True,
    constrained_layout=True
)

make_panel(
    ax=axes[0],
    y_col="signed_tc_error",
    y_label="Error",
    model=signed_fe,
    point_color=TOP_POINT_COLOR
)

make_panel(
    ax=axes[1],
    y_col="absolute_tc_error",
    y_label="Absolute Error",
    model=abs_fe,
    point_color=BOTTOM_POINT_COLOR
)

axes[0].set_xlabel("Confidence Indicator")
axes[1].set_xlabel("Confidence Indicator")

axes[0].axhline(0, color="gray", linestyle="--", linewidth=1.2)
axes[1].axhline(0, color="gray", linestyle="--", linewidth=1.2)

# Do not use bbox_inches="tight" if you need exact final PNG aspect ratio
plt.savefig(OUTPUT_PNG, dpi=300)
plt.show()

print(f"\nSaved plot: {OUTPUT_PNG}")
print(f"Saved panel: {OUTPUT_PANEL}")