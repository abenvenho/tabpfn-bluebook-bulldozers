"""As is, where is — TabPFN-3.5 on the Blue Book for Bulldozers (Streamlit app).

Run from the repository root:

    streamlit run app/streamlit_app.py

Three tabs:

1. The test — the bet, the pre-registered verdicts and the figures, read from results/.
2. Jan–Apr 2012 sales — every validation sale: realized price, TabPFN-3.5 (raw table,
   all 401,125 training sales) with its 80% interval, and the engineered LightGBM.
   Read from the versioned prediction files; no API call. Machine attributes appear
   when data/prepared/ exists (the Kaggle files are not redistributed; DATA_NOTICE.md).
3. Price a machine as is — edit a sale's raw attributes (dirty values welcome) and ask
   TabPFN-3.5 live through the Prior Labs API, with the 50,000 most recent training
   sales as context (the 50k-recent configuration of H3). Needs data/prepared/ and your
   own Prior Labs token. The server's cost quote comes before any paid call, and the
   usage counter is read before and after it.

Nothing here changes a pre-registered verdict (results/verdicts.md). Breakdowns computed
in the app (by product group, by data-quality slice) are exploratory.

Testing without the API: BLUEBOOK_APP_MOCK=1 streamlit run app/streamlit_app.py swaps
TabPFN for the smoke test's stand-in predictor and says so on every result.
"""
import contextlib
import os
import re
import sys
from pathlib import Path

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.features import to_matrix  # noqa: E402
from src.schema import (DATE_COL, DATE_PARSED, ID_COL, NUMERIC_COLS,  # noqa: E402
                        RAW_PREDICTORS, TARGET)
from src.tabpfn_runner import (QCOLS, QUANTILES, build_arm, predict_api,  # noqa: E402
                               predict_mock, select_context)

RESULTS = ROOT / "results"
PREPARED = ROOT / "data" / "prepared"
REPO_URL = "https://github.com/abenvenho/tabpfn-bluebook-bulldozers"
LIVE_CONTEXT = "50000"
MOCK = os.environ.get("BLUEBOOK_APP_MOCK") == "1"

BLUE, ORANGE, INK, MUTED, GRID = "#2A78D6", "#EB6834", "#1B222C", "#5C6570", "#E4DED2"

st.set_page_config(page_title="As is, where is — TabPFN-3.5 on the Blue Book",
                   page_icon="🚜", layout="wide")
alt.data_transformers.disable_max_rows()


# ----------------------------------------------------------------------------- data

def rmsle(y, p) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    return float(np.sqrt(np.mean((np.log1p(p) - np.log1p(y)) ** 2)))


def money(x) -> str:
    return "—" if x is None or pd.isna(x) else f"US$ {x:,.0f}"


@st.cache_data
def load_metrics() -> dict:
    out = {}
    for f in (RESULTS / "metrics").glob("*.json"):
        out[f.stem] = pd.read_json(f, typ="series").to_dict()
    return out


@st.cache_data
def load_preds() -> pd.DataFrame:
    """One row per validation sale: realized price and the versioned predictions."""
    p = pd.read_csv(RESULTS / "preds" / "primary_raw.csv").rename(columns={"pred": "tabpfn"})
    lgbm = pd.read_csv(RESULTS / "preds" / "lgbm_engineered.csv")[[ID_COL, "pred"]]
    r50 = pd.read_csv(RESULTS / "preds" / "tabpfn_base_raw_valid_50000_recent.csv")
    r50 = r50[[ID_COL, "pred"] + QCOLS].rename(
        columns={"pred": "tabpfn_50k", **{c: f"{c}_50k" for c in QCOLS}})
    df = (p.merge(lgbm.rename(columns={"pred": "lgbm"}), on=ID_COL, validate="one_to_one")
          .merge(r50, on=ID_COL, validate="one_to_one"))
    df["inside80"] = (df["y_true"] >= df["q10"]) & (df["y_true"] <= df["q90"])
    return df


@st.cache_data
def load_valid():
    f = PREPARED / "valid.parquet"
    return pd.read_parquet(f) if f.exists() else None


@st.cache_resource
def load_live_context():
    """The 50,000 most recent training sales, selected exactly as in the H3 run."""
    f = PREPARED / "train.parquet"
    if not f.exists():
        return None
    train = pd.read_parquet(f)
    ctx = select_context(train, LIVE_CONTEXT, "recent").copy()
    del train
    return ctx


def parse_verdicts() -> pd.DataFrame:
    text = (RESULTS / "verdicts.md").read_text(encoding="utf-8").splitlines()
    rows, i = [], 0
    while i < len(text):
        m = re.match(r"^## (H\d)(?: \(([^)]+)\))? — (.+)$", text[i])
        if m:
            j = i + 1
            while j < len(text) and not text[j].strip():
                j += 1
            measured = text[j].strip() if j < len(text) else ""
            rows.append({"Hypothesis": m.group(1), "Part": m.group(2) or "",
                         "Verdict": m.group(3).strip(), "Measured": measured})
            i = j
        i += 1
    return pd.DataFrame(rows)


HYPOTHESES = {
    "H1": "Raw TabPFN-3.5 is not worse than the engineered LightGBM",
    "H2": "Cleaning, or the Machine Appendix, does not improve TabPFN-3.5 by more than 0.005",
    "H3": "At equal context size, the most recent sales beat a random sample",
    "H4": "The 80% interval holds four months ahead (coverage 75–85%)",
    "H5": "The primary run scores in the 2013 top 10% (leaderboard-mirror split)",
}


# ----------------------------------------------------------------------------- charts

def interval_chart(rows: pd.DataFrame, realized=None) -> alt.Chart:
    """One horizontal band per row: q10–q90 light, q25–q75 dark, mean as a dot.

    rows: columns label, q10, q25, q75, q90, mean (any of the q's may be NaN).
    """
    base = alt.Chart(rows).encode(y=alt.Y("label:N", title=None, sort=None,
                                          axis=alt.Axis(labelLimit=320, labelFontSize=13)))
    x = alt.X("q10:Q", title="Price (US$)", scale=alt.Scale(zero=False, nice=True),
              axis=alt.Axis(format="$,.0f", gridColor=GRID))
    outer = base.mark_rule(color="#BCD3F1", strokeWidth=14).encode(x=x, x2="q90:Q")
    inner = base.mark_rule(color=BLUE, strokeWidth=14).encode(x="q25:Q", x2="q75:Q")
    tip = [alt.Tooltip("label:N", title=""), alt.Tooltip("mean:Q", title="mean", format="$,.0f"),
           alt.Tooltip("q10:Q", title="q10", format="$,.0f"),
           alt.Tooltip("q90:Q", title="q90", format="$,.0f")]
    dot = base.mark_point(filled=True, size=140, color=INK, stroke="#FBF9F4",
                          strokeWidth=2).encode(x="mean:Q", tooltip=tip)
    layers = [outer, inner, dot]
    if realized is not None:
        rdf = rows[["label"]].assign(realized=float(realized))
        layers.append(alt.Chart(rdf).mark_tick(color=ORANGE, thickness=3, size=30).encode(
            y=alt.Y("label:N", sort=None, title=None), x="realized:Q",
            tooltip=[alt.Tooltip("realized:Q", title="realized price", format="$,.0f")]))
    h = 46 * len(rows) + 30
    return alt.layer(*layers).properties(height=h).configure_view(strokeWidth=0)


def scatter_chart(df: pd.DataFrame) -> alt.Chart:
    d = df.assign(outcome=np.where(df["inside80"], "inside the 80% interval",
                                   "outside the 80% interval"))
    lo = float(min(d["y_true"].min(), d["tabpfn"].min())) * 0.9
    hi = float(max(d["y_true"].max(), d["tabpfn"].max())) * 1.1
    scale = alt.Scale(type="log", domain=[lo, hi], nice=False)
    ticks = [5000, 10000, 20000, 50000, 100000]
    tips = [alt.Tooltip(f"{ID_COL}:N", title="SalesID"),
            alt.Tooltip("y_true:Q", title="realized", format="$,.0f"),
            alt.Tooltip("tabpfn:Q", title="TabPFN-3.5", format="$,.0f"),
            alt.Tooltip("lgbm:Q", title="LightGBM", format="$,.0f")]
    if "fiModelDesc" in d.columns:
        tips.insert(1, alt.Tooltip("fiModelDesc:N", title="model"))
    pts = alt.Chart(d).mark_circle(size=18, opacity=0.45).encode(
        x=alt.X("tabpfn:Q", title="TabPFN-3.5 prediction (US$, log scale)", scale=scale,
                axis=alt.Axis(format="$,.0f", gridColor=GRID, values=ticks)),
        y=alt.Y("y_true:Q", title="Realized price (US$, log scale)", scale=scale,
                axis=alt.Axis(format="$,.0f", gridColor=GRID, values=ticks)),
        color=alt.Color("outcome:N", title=None,
                        scale=alt.Scale(domain=["inside the 80% interval",
                                                "outside the 80% interval"],
                                        range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient="top")),
        tooltip=tips)
    diag = alt.Chart(pd.DataFrame({"v": [lo, hi]})).mark_line(
        color=MUTED, strokeDash=[6, 4], strokeWidth=1.5).encode(x="v:Q", y="v:Q")
    return (diag + pts).properties(height=520).configure_view(strokeWidth=0)


def group_chart(tbl: pd.DataFrame) -> alt.Chart:
    long = tbl.melt(id_vars=["Product group"], value_vars=["TabPFN-3.5", "LightGBM"],
                    var_name="Model", value_name="RMSLE")
    order = tbl.sort_values("TabPFN-3.5")["Product group"].tolist()
    rule = alt.Chart(tbl).mark_rule(color="#C9C2B3", strokeWidth=2).encode(
        y=alt.Y("Product group:N", sort=order, title=None),
        x="TabPFN-3.5:Q", x2="LightGBM:Q")
    dots = alt.Chart(long).mark_point(filled=True, size=160, stroke="#FBF9F4",
                                      strokeWidth=2).encode(
        y=alt.Y("Product group:N", sort=order, title=None, axis=alt.Axis(labelLimit=220)),
        x=alt.X("RMSLE:Q", scale=alt.Scale(zero=False), axis=alt.Axis(gridColor=GRID),
                title="RMSLE, Jan–Apr 2012 (lower is better)"),
        color=alt.Color("Model:N", scale=alt.Scale(domain=["TabPFN-3.5", "LightGBM"],
                                                   range=[BLUE, ORANGE]),
                        legend=alt.Legend(orient="top", title=None)),
        shape=alt.Shape("Model:N", scale=alt.Scale(domain=["TabPFN-3.5", "LightGBM"],
                                                   range=["circle", "square"]),
                        legend=None),
        tooltip=["Product group:N", "Model:N", alt.Tooltip("RMSLE:Q", format=".4f")])
    return (rule + dots).properties(height=40 * len(tbl) + 40).configure_view(strokeWidth=0)


# ----------------------------------------------------------------------------- helpers

def show(v) -> str:
    """How a raw value is displayed and edited: empty for missing, else as stored."""
    if v is None or (isinstance(v, float) and np.isnan(v)) or (v is pd.NaT):
        return ""
    if isinstance(v, (np.integer, int)):
        return str(int(v))
    if isinstance(v, (np.floating, float)):
        return repr(float(v))
    return str(v)


def dirt_flags(row: pd.Series) -> list:
    flags = []
    if row.get("YearMade") == 1000:
        flags.append("YearMade recorded as 1000")
    h = row.get("MachineHoursCurrentMeter")
    if pd.isna(h):
        flags.append("hour meter missing")
    elif h == 0:
        flags.append("hour meter zero")
    empty = int(sum(pd.isna(row.get(c)) for c in RAW_PREDICTORS))
    flags.append(f"{empty} of {len(RAW_PREDICTORS)} columns empty")
    return flags


def apply_edits(orig: pd.Series, edited: dict) -> tuple:
    """Copy of the sale with only the changed fields replaced. Unchanged fields keep
    their stored values and types, so an unedited sale reaches the API exactly as in
    the batch run."""
    row = orig.copy()
    changed = []
    for col, new in edited.items():
        new = "" if new is None else str(new).strip()
        if new == show(orig[col]):
            continue
        changed.append(col)
        if new == "":
            row[col] = np.nan
        elif col in NUMERIC_COLS:
            row[col] = pd.to_numeric(new, errors="coerce")
        else:
            row[col] = new
    if DATE_COL in changed:
        row[DATE_PARSED] = pd.to_datetime(row[DATE_COL], errors="coerce")
    return row, changed


def as_frame(row: pd.Series) -> pd.DataFrame:
    """One-row frame with the column types the pipeline expects."""
    df = pd.DataFrame([row])
    for c in df.columns:
        if c == DATE_PARSED:
            df[c] = pd.to_datetime(df[c], errors="coerce")
        elif c in NUMERIC_COLS or c in (TARGET, ID_COL):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = df[c].astype(object)
    return df


@contextlib.contextmanager
def token_env(token: str):
    """predict_api reads TABPFN_TOKEN; set it for the duration of one call."""
    old = os.environ.get("TABPFN_TOKEN")
    os.environ["TABPFN_TOKEN"] = token
    try:
        yield
    finally:
        if old is None:
            os.environ.pop("TABPFN_TOKEN", None)
        else:
            os.environ["TABPFN_TOKEN"] = old


def usage_now(token: str):
    try:
        import tabpfn_client
        from tabpfn_client.client import ServiceClient
        tabpfn_client.set_access_token(token)
        return ServiceClient.get_api_usage(token).get("current_usage")
    except Exception:
        return None


def quote_tokens(token: str, Xc, Xe):
    import tabpfn_client
    tabpfn_client.set_access_token(token)
    return int(tabpfn_client.estimate_cost(Xc, Xe).estimated_cost)


def live_predict(arm: str, ctx: pd.DataFrame, row: pd.DataFrame, token: str) -> dict:
    Xc, cols = to_matrix(build_arm(ctx, arm, None))
    Xe, _ = to_matrix(build_arm(row, arm, None), feature_cols=cols)
    yc = np.log1p(ctx[TARGET].to_numpy())
    if MOCK:
        mean, qs, info = predict_mock(Xc, yc, Xe)
    else:
        with token_env(token):
            mean, qs, info = predict_api(Xc, yc, Xe, "base", None)
    out = {"mean": float(np.expm1(mean[0])), "predictor": info.get("predictor")}
    for q, c in zip(QUANTILES, QCOLS):
        out[c] = float(np.expm1(qs[q][0])) if q in qs else np.nan
    return out


def matrices(arm: str, ctx: pd.DataFrame, row: pd.DataFrame):
    Xc, cols = to_matrix(build_arm(ctx, arm, None))
    Xe, _ = to_matrix(build_arm(row, arm, None), feature_cols=cols)
    return Xc, Xe


# ----------------------------------------------------------------------------- page

metrics = load_metrics()
preds = load_preds()
valid = load_valid()
sales = preds if valid is None else valid.drop(columns=[TARGET]).merge(
    preds, on=ID_COL, validate="one_to_one")

st.title("As is, where is")
st.markdown(
    "**Does a tabular foundation model need the table cleaned first?** "
    "TabPFN-3.5, zero-shot, on the raw Kaggle *Blue Book for Bulldozers* auction table, "
    "against a LightGBM given 2013-style cleaning and feature engineering. "
    f"Trained on sales through 2011, scored on January–April 2012. [Repository]({REPO_URL})")
if MOCK:
    st.warning("MOCK MODE: live prices in tab 3 come from a local stand-in model, "
               "not from TabPFN-3.5. Unset BLUEBOOK_APP_MOCK for real calls.", icon="⚠️")

# Sidebar: one sale drives tabs 2 and 3.
with st.sidebar:
    st.header("Pick a sale")
    st.caption("11,573 auction sales, January–April 2012")
    pool = sales
    if valid is not None:
        groups = ["All product groups"] + sorted(pool["ProductGroupDesc"].dropna().unique())
        g = st.selectbox("Product group", groups)
        if g != groups[0]:
            pool = pool[pool["ProductGroupDesc"] == g]
        dirt = st.selectbox("Data quality", ["Any row", "YearMade recorded as 1000",
                                             "Hour meter missing", "Hour meter zero",
                                             "YearMade 1000 and hour meter missing"],
                            index=4)
        if dirt == "YearMade recorded as 1000":
            pool = pool[pool["YearMade"] == 1000]
        elif dirt == "Hour meter missing":
            pool = pool[pool["MachineHoursCurrentMeter"].isna()]
        elif dirt == "Hour meter zero":
            pool = pool[pool["MachineHoursCurrentMeter"] == 0]
        elif dirt.startswith("YearMade 1000 and"):
            pool = pool[(pool["YearMade"] == 1000) & pool["MachineHoursCurrentMeter"].isna()]
        pool = pool.sort_values(DATE_PARSED)
    outcome = st.selectbox("Outcome", ["Any", "Inside the 80% interval",
                                       "Outside the 80% interval"])
    if outcome.startswith("Inside"):
        pool = pool[pool["inside80"]]
    elif outcome.startswith("Outside"):
        pool = pool[~pool["inside80"]]
    if len(pool) == 0:
        st.warning("No sale matches these filters; showing all sales.")
        pool = sales.sort_values(DATE_PARSED) if valid is not None else sales
    st.caption(f"{len(pool):,} sales match")

    def label(sid):
        r = pool.loc[pool[ID_COL] == sid].iloc[0]
        if valid is None:
            return f"{sid}"
        return f"{sid} · {r['fiModelDesc']} · {r['ProductGroupDesc']} · {r[DATE_COL].split()[0]}"

    sid = st.selectbox("Sale", pool[ID_COL].tolist(), format_func=label)
    sale = sales.loc[sales[ID_COL] == sid].iloc[0]
    st.divider()
    st.caption("Tabs 2 and 3 follow this choice. The default filter lists the dirtiest "
               "rows first: no year of manufacture and no hour reading.")

tab1, tab2, tab3 = st.tabs(["The test", "Jan–Apr 2012 sales", "Price a machine as is"])

# ----------------------------------------------------------------------------- tab 1
with tab1:
    pr, lg = metrics["primary_raw"], metrics["lgbm_engineered"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("TabPFN-3.5, raw table, zero-shot", f"{pr['rmsle']:.4f}", help="RMSLE, lower is better")
    c2.metric("LightGBM, 2013-style cleaning", f"{lg['rmsle']:.4f}", help="RMSLE, lower is better")
    c3.metric("Difference (paired)", f"{pr['rmsle'] - lg['rmsle']:+.5f}",
              help="Paired bootstrap 95% CI in results/verdicts.md")
    c4.metric("80% interval coverage", f"{pr['coverage80']:.1%}", help="Pre-registered band 75–85%")

    st.subheader("Five hypotheses, frozen in git before the first run")
    v = parse_verdicts()
    lines = ["| | Pre-registered statement | Measured (paired bootstrap, 95% CI) | Verdict |",
             "|---|---|---|---|"]
    for _, r in v.iterrows():
        measured = r["Measured"].split("; refuted iff")[0].replace(" .. ", " to ")
        part = f" ({r['Part']})" if r["Part"] else ""
        verdict = r["Verdict"].lower()
        verdict = "**REFUTED**" if verdict == "refuted" else verdict
        lines.append(f"| **{r['Hypothesis']}**{part} | {HYPOTHESES[r['Hypothesis']]} | "
                     f"{measured} | {verdict} |")
    st.markdown("\n".join(lines))
    st.caption("Written by src/compare.py from the result files. Differences are paired "
               "bootstraps over the 11,573 sales (10,000 resamples, 95% CI); H1 and H2 use "
               "a 0.005 margin. H5 compares across test sets, so it is indicative, not a "
               "leaderboard rank.")

    st.subheader("Figures")
    f1, f2 = st.columns(2)
    f1.image(str(RESULTS / "figures" / "rmsle_by_run.png"),
             caption="Every configuration against the baselines", width="stretch")
    f2.image(str(RESULTS / "figures" / "context_h3.png"),
             caption="H3 — refuted: recency helps at 50k, random wins at 200k", width="stretch")
    f3, f4 = st.columns(2)
    f3.image(str(RESULTS / "figures" / "calibration.png"),
             caption="H4 — quantile calibration four months ahead", width="stretch")
    f4.image(str(RESULTS / "figures" / "pred_vs_actual.png"),
             caption="Predicted against realized prices", width="stretch")
    st.markdown(f"Full write-up: [README]({REPO_URL}#readme) · "
                f"[paper]({REPO_URL}/blob/main/paper/bluebook_tabpfn_paper.pdf) · "
                f"[pre-registration]({REPO_URL}/blob/main/PREREGISTRATION.md)")

# ----------------------------------------------------------------------------- tab 2
with tab2:
    st.subheader(f"Sale {sid}" + (f" — {sale['fiModelDesc']}, {sale['ProductGroupDesc']}"
                                  if valid is not None else ""))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Realized price", money(sale["y_true"]))
    c2.metric("TabPFN-3.5 (all sales, raw)", money(sale["tabpfn"]),
              f"{sale['tabpfn'] / sale['y_true'] - 1:+.1%} vs realized", delta_color="off")
    c3.metric("LightGBM (engineered)", money(sale["lgbm"]),
              f"{sale['lgbm'] / sale['y_true'] - 1:+.1%} vs realized", delta_color="off")
    c4.metric("80% interval (US$)", f"{sale['q10']:,.0f} – {sale['q90']:,.0f}",
              "realized inside" if sale["inside80"] else "realized outside", delta_color="off")
    rows = pd.DataFrame([{"label": "TabPFN-3.5, all 401,125 sales", "mean": sale["tabpfn"],
                          "q10": sale["q10"], "q25": sale["q25"], "q75": sale["q75"],
                          "q90": sale["q90"]}])
    st.altair_chart(interval_chart(rows, realized=sale["y_true"]), width="stretch")
    st.caption("Light band: 80% interval (q10–q90). Dark band: q25–q75. Dot: mean prediction. "
               "Orange mark: realized auction price.")

    if valid is not None:
        st.markdown("**What TabPFN-3.5 received for this sale**, exactly as in the CSV: "
                    + " · ".join(dirt_flags(sale)))
        raw = pd.DataFrame({"Column": RAW_PREDICTORS,
                            "Value as received": [show(sale[c]) or "(empty)" for c in RAW_PREDICTORS]})
        with st.expander("All 51 predictor columns", expanded=False):
            st.dataframe(raw, hide_index=True, width="stretch", height=420)
    else:
        st.info("Machine attributes need the Kaggle files: put them in data/raw/ and run "
                "scripts/01_prepare.py (see DATA_NOTICE.md). Prices and predictions above come "
                "from the versioned result files.")

    st.divider()
    st.subheader("All 11,573 sales")
    a1, a2, a3 = st.columns(3)
    a1.metric("RMSLE, TabPFN-3.5", f"{rmsle(preds['y_true'], preds['tabpfn']):.4f}")
    a2.metric("RMSLE, LightGBM", f"{rmsle(preds['y_true'], preds['lgbm']):.4f}")
    a3.metric("Realized inside the 80% interval", f"{preds['inside80'].mean():.1%}")
    st.altair_chart(scatter_chart(sales), width="stretch")

    if valid is not None:
        st.markdown("**Exploratory breakdowns** (computed here, not pre-registered)")
        g = (sales.groupby("ProductGroupDesc")
             .apply(lambda d: pd.Series({
                 "Sales": len(d),
                 "TabPFN-3.5": rmsle(d["y_true"], d["tabpfn"]),
                 "LightGBM": rmsle(d["y_true"], d["lgbm"]),
                 "80% coverage": d["inside80"].mean()}), include_groups=False)
             .reset_index().rename(columns={"ProductGroupDesc": "Product group"}))
        g["Sales"] = g["Sales"].astype(int)
        b1, b2 = st.columns([1, 1])
        b1.altair_chart(group_chart(g), width="stretch")
        b2.dataframe(g, hide_index=True, width="stretch",
                     column_config={"TabPFN-3.5": st.column_config.NumberColumn(format="%.4f"),
                                    "LightGBM": st.column_config.NumberColumn(format="%.4f"),
                                    "80% coverage": st.column_config.NumberColumn(format="percent")})

        slices = {
            "All sales": np.ones(len(sales), bool),
            "YearMade recorded as 1000": (sales["YearMade"] == 1000).to_numpy(),
            "Hour meter missing": sales["MachineHoursCurrentMeter"].isna().to_numpy(),
            "Hour meter zero": (sales["MachineHoursCurrentMeter"] == 0).to_numpy(),
            "Both year and hours unknown": ((sales["YearMade"] == 1000)
                                            & sales["MachineHoursCurrentMeter"].isna()).to_numpy(),
        }
        q = pd.DataFrame([{"Rows": k, "Sales": int(m.sum()),
                           "TabPFN-3.5": rmsle(sales["y_true"][m], sales["tabpfn"][m]),
                           "LightGBM": rmsle(sales["y_true"][m], sales["lgbm"][m]),
                           "80% coverage": sales["inside80"][m].mean()}
                          for k, m in slices.items()])
        st.markdown("By data quality of the row (LightGBM saw YearMade 1000 and zero hours "
                    "as missing; TabPFN-3.5 saw them as recorded):")
        st.dataframe(q, hide_index=True, width="stretch",
                     column_config={"TabPFN-3.5": st.column_config.NumberColumn(format="%.4f"),
                                    "LightGBM": st.column_config.NumberColumn(format="%.4f"),
                                    "80% coverage": st.column_config.NumberColumn(format="percent")})
        st.caption("Point estimates on subsets, without paired tests. The pre-registered "
                   "comparison is H1, over all sales.")

# ----------------------------------------------------------------------------- tab 3
with tab3:
    st.markdown(
        "Edit the sale's raw attributes, dirty values welcome, and ask TabPFN-3.5 for a "
        "price through the Prior Labs API. Context: the **50,000 most recent training sales**, "
        "raw table, zero-shot (the 50k-recent configuration of H3, RMSLE 0.2263 over all "
        "11,573 sales). The server quotes the cost first; each pricing makes two calls, "
        "quantiles then mean, as in the study.")
    ctx = load_live_context() if valid is not None else None
    if ctx is None:
        st.info("This tab needs data/prepared/: put the Kaggle files in data/raw/ and run "
                "scripts/01_prepare.py (see DATA_NOTICE.md).")
        st.stop()

    env_token = os.environ.get("TABPFN_TOKEN", "")
    if MOCK:
        token = "mock"
    elif env_token:
        token = env_token
        st.caption("Using the Prior Labs token from the TABPFN_TOKEN environment variable.")
    else:
        token = st.text_input("Your Prior Labs API token", type="password",
                              help="Kept in this session only; never written to disk.")

    orig = valid.loc[valid[ID_COL] == sid].iloc[0]
    editor_df = pd.DataFrame({"Column": RAW_PREDICTORS,
                              "Value": [show(orig[c]) for c in RAW_PREDICTORS]})
    left, right = st.columns([3, 2])
    with left:
        st.markdown(f"**Sale {sid}** as the template. Empty cell = missing value.")
        edited = st.data_editor(
            editor_df, hide_index=True, width="stretch", height=430, key=f"edit_{sid}",
            disabled=["Column"],
            column_config={"Value": st.column_config.TextColumn("Value (edit me)")})
    row_s, changed = apply_edits(orig, dict(zip(edited["Column"], edited["Value"])))
    # Unedited: the very row of the batch run. Edited: the same row with the changes.
    row = valid.loc[valid[ID_COL] == sid] if not changed else as_frame(row_s)
    with right:
        st.markdown("**This machine, as TabPFN-3.5 will receive it:** "
                    + " · ".join(dirt_flags(row_s)))
        if changed:
            st.markdown("Edited: " + ", ".join(f"`{c}`" for c in changed))
        else:
            st.markdown("Unedited: the same row as in the batch run, so the live answer can "
                        "be compared with the versioned 50k prediction.")
        also_clean = st.checkbox(
            "Also price the 2013-cleaned version of this row (same 50k context, cleaned). "
            "Doubles the cost.", value=True)
        arms = ["raw", "clean"] if also_clean else ["raw"]
        key = (sid, tuple(changed), tuple(show(row_s[c]) for c in changed), tuple(arms))
        ready = bool(token)

        if st.button("1 · Quote the cost (free)", disabled=not ready, width="stretch"):
            try:
                if MOCK:
                    st.session_state["quote"] = (key, {a: None for a in arms})
                else:
                    with st.spinner("Asking the server for a quote…"):
                        qd = {a: quote_tokens(token, *matrices(a, ctx, row)) for a in arms}
                    st.session_state["quote"] = (key, qd)
            except Exception as e:
                st.error(f"Quote failed: {e}")
        quote = st.session_state.get("quote")
        quoted = quote is not None and quote[0] == key
        if quoted:
            qd = quote[1]
            if MOCK:
                st.caption("Mock mode: no quote.")
            else:
                total = sum(2 * v for v in qd.values())
                st.markdown("Server quote per call: " + ", ".join(
                    f"{a} {v:,} tokens" for a, v in qd.items())
                    + f". This pricing makes {2 * len(qd)} calls: about **{total:,} tokens**.")

        if st.button("2 · Price it with TabPFN-3.5", type="primary",
                     disabled=not (ready and quoted), width="stretch"):
            before = None if MOCK else usage_now(token)
            res = {}
            try:
                with st.spinner("Calling TabPFN-3.5 (50,000 sales of context)…"):
                    for a in arms:
                        res[a] = live_predict(a, ctx, row, token)
            except Exception as e:
                st.error(f"Call failed: {e}")
            after = None if MOCK else usage_now(token)
            if res:
                spent = (after - before) if (before is not None and after is not None) else None
                st.session_state["live"] = (key, res, spent)
                st.session_state.setdefault("history", []).append({
                    "SalesID": sid, "edited": ", ".join(changed) or "—",
                    "raw (US$)": round(res["raw"]["mean"]),
                    "cleaned (US$)": round(res["clean"]["mean"]) if "clean" in res else None,
                    "tokens measured": spent})
        if not ready:
            st.caption("Enter your token to enable the buttons.")
        elif not quoted:
            st.caption("Quote first: the call button unlocks after the quote.")

    live = st.session_state.get("live")
    if live is not None and live[0][0] == sid:
        key_l, res, spent = live
        st.divider()
        tag = " (MOCK — not TabPFN)" if MOCK else ""
        rows = []
        for a in res:
            r = res[a]
            rows.append({"label": ("As is (raw)" if a == "raw" else "Cleaned, 2013-style") + tag,
                         "mean": r["mean"], "q10": r["q10"], "q25": r["q25"],
                         "q75": r["q75"], "q90": r["q90"]})
        if not key_l[1]:  # unedited: show the versioned batch answers too
            rows.append({"label": "Versioned batch run, 50k recent", "mean": sale["tabpfn_50k"],
                         "q10": sale["q10_50k"], "q25": sale["q25_50k"],
                         "q75": sale["q75_50k"], "q90": sale["q90_50k"]})
        st.altair_chart(interval_chart(pd.DataFrame(rows),
                                       realized=None if key_l[1] else sale["y_true"]),
                        width="stretch")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("As is (raw)" + tag, money(res["raw"]["mean"]),
                  f"80%: US$ {res['raw']['q10']:,.0f} – {res['raw']['q90']:,.0f}",
                  delta_color="off")
        if "clean" in res:
            m2.metric("Cleaned, 2013-style" + tag, money(res["clean"]["mean"]),
                      f"{res['clean']['mean'] / res['raw']['mean'] - 1:+.1%} vs as is",
                      delta_color="off")
        if not key_l[1]:
            m3.metric("Realized price", money(sale["y_true"]),
                      f"versioned 50k batch: US$ {sale['tabpfn_50k']:,.0f}",
                      delta_color="off")
        m4.metric("Tokens measured", "—" if spent is None else f"{spent:,}",
                  help="Difference of the API usage counter before and after the calls")
        st.caption("Exploratory: single live predictions, not part of the pre-registered test.")

    if st.session_state.get("history"):
        with st.expander("This session's live prices"):
            st.dataframe(pd.DataFrame(st.session_state["history"]), hide_index=True,
                         width="stretch")
