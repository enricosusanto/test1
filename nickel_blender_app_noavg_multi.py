import io, math
import pandas as pd
import numpy as np
import streamlit as st
from io import BytesIO

# ---------------- Basic settings ----------------
st.set_page_config(page_title="Nickel Blending Optimizer — Multi/Scenario", layout="wide")
st.title("🛠️ Nickel Blending Optimizer — Multi-Cargo + Scenario Builder (Ni ≥ floor)")

# --------------- Helpers: IO & normalize ---------------
def detect_header_and_delim(file_bytes):
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    delim = ";" if lines and lines[0].count(";") >= 3 else ","
    header_idx = 0
    tokens = ["lot", "wmt", "wet", "ni", "fe", "sio2", "mgo", "moist"]
    for i, line in enumerate(lines[:100]):
        low = line.lower()
        if any(tok in low for tok in tokens) and line.count(delim) >= 2:
            header_idx = i
            break
    return delim, header_idx, lines

def read_dataframe(upload):
    content = upload.getvalue()
    delim, header_idx, lines = detect_header_and_delim(content)
    text = "\n".join(lines[header_idx:])
    df = pd.read_csv(io.StringIO(text), sep=delim, engine="python")
    df.columns = [str(c).strip() for c in df.columns]
    return df, delim, header_idx

def map_columns(df, lot_col_opt=None, wmt_col_opt=None):
    cols = list(df.columns)
    def find_col(cands):
        for cand in cands:
            for c in cols:
                if cand == str(c).lower():
                    return c
            for c in cols:
                if cand in str(c).lower():
                    return c
        return None
    lot_col = lot_col_opt if lot_col_opt else None
    wmt_col = wmt_col_opt if wmt_col_opt else None
    ni_col  = find_col(["ni(%)", "ni %", "ni", "nickel", "ni grade"])
    fe_col  = find_col(["fe(%)", "fe %", "fe"])
    sio2_col= find_col(["sio2(%)", "sio2 %", "sio2"])
    mgo_col = find_col(["mgo(%)", "mgo %", "mgo"])
    moist_col = find_col(["moisture(%)", "moisture %", "moisture", "h2o", "h2o%"])
    return {"LotID": lot_col, "WMT_available": wmt_col, "Ni": ni_col,
            "Fe": fe_col, "SiO2": sio2_col, "MgO": mgo_col, "Moisture": moist_col}

def normalize_df(df, mapping, header_position_map=None):
    norm = pd.DataFrame()
    if mapping["LotID"] is None and header_position_map and "A" in header_position_map:
        idx = header_position_map["A"]
        if idx < len(df.columns): mapping["LotID"] = df.columns[idx]
    if mapping["WMT_available"] is None and header_position_map and "K" in header_position_map:
        idx = header_position_map["K"]
        if idx < len(df.columns): mapping["WMT_available"] = df.columns[idx]
    for std, src in mapping.items():
        if src is not None and src in df.columns:
            norm[std] = df[src]
    if "LotID" not in norm.columns:
        norm["LotID"] = df.index.astype(str)
    for c in ["WMT_available", "Ni", "Fe", "SiO2", "MgO", "Moisture"]:
        if c in norm.columns:
            norm[c] = pd.to_numeric(
                norm[c].astype(str)
                      .str.replace(",", ".", regex=False)
                      .str.replace(r"[^0-9\.\-]", "", regex=True),
                errors="coerce"
            )
    core = [c for c in ["LotID", "WMT_available", "Ni"] if c in norm.columns]
    if core:
        norm = norm.dropna(subset=core, how="any")
    if "WMT_available" in norm.columns:
        norm = norm[norm["WMT_available"] > 0]
    return norm

# --------------- Optimization core (no-avg objective) ---------------
def package_solution(lots, x_map, T, G, eps):
    rows = []
    total_wmt = 0.0
    numer = 0.0
    for lot in lots:
        x = max(0.0, x_map.get(lot["id"], 0.0))
        if x > 0:
            rows.append({
                "LotID": lot["id"],
                "WMT_to_load": round(x, 2),
                "Pct_Share_%": round(100.0 * x / T, 2),
                "Ni_%": lot["ni"],
            })
            total_wmt += x
            numer += x * lot["ni"]
    rows_df = pd.DataFrame(rows).sort_values(by="WMT_to_load", ascending=False)
    avg_ni = numer / total_wmt if total_wmt > 0 else float("nan")
    lots_used = (rows_df["WMT_to_load"] > 0).sum() if len(rows_df) else 0
    kpis = {
        "Total_WMT": round(total_wmt, 2),
        "Weighted_Ni_%": round(avg_ni, 4) if not math.isnan(avg_ni) else None,
        "Lots_Used": int(lots_used),
        "Floor_or_Target_%": G,
        "Tolerance_%": eps if eps is not None else None,
    }
    return rows_df, kpis

def optimize_single_cargo(norm_df, T, G, eps, x_min, prefer_min_lots=True):
    lots = []
    for _, r in norm_df.iterrows():
        lots.append({"id": str(r["LotID"]), "avail": float(r["WMT_available"]), "ni": float(r["Ni"])})
    try:
        import pulp as pl
        M = {lot["id"]: min(lot["avail"], T) for lot in lots}
        prob = pl.LpProblem("blend_min_lots", pl.LpMinimize)
        x = {lot["id"]: pl.LpVariable(f"x_{i}", lowBound=0) for i, lot in enumerate(lots)}
        y = {lot["id"]: pl.LpVariable(f"y_{i}", lowBound=0, upBound=1, cat=pl.LpBinary) for i, lot in enumerate(lots)}
        prob += pl.lpSum(x[lot["id"]] for lot in lots) == T
        for lot in lots:
            prob += x[lot["id"]] <= lot["avail"]
            prob += x[lot["id"]] <= y[lot["id"]] * M[lot["id"]]
            prob += x[lot["id"]] >= y[lot["id"]] * x_min
        if eps is None:
            prob += pl.lpSum(x[lot["id"]] * lot["ni"] for lot in lots) >= (G) * T
        else:
            prob += pl.lpSum(x[lot["id"]] * lot["ni"] for lot in lots) >= (G - eps) * T
            prob += pl.lpSum(x[lot["id"]] * lot["ni"] for lot in lots) <= (G + eps) * T
        prob.setObjective(pl.lpSum(y[lot["id"]]) if prefer_min_lots else 0)
        prob.solve(pl.PULP_CBC_CMD(msg=False))
        x_map = {lot["id"]: float(pl.value(x[lot["id"]]) or 0) for lot in lots}
        rows_df, kpis = package_solution(lots, x_map, T, G, eps)
        return rows_df, kpis, "PuLP-CBC"
    except Exception as e:
        # Greedy fallback
        remaining = T
        x_map = {}
        if eps is None:
            candidates = sorted([l for l in lots if l["avail"] >= x_min], key=lambda l: (-1 if l["ni"] >= G else 1, -l["ni"]))
        else:
            candidates = sorted([l for l in lots if l["avail"] >= x_min], key=lambda l: (abs(l["ni"] - G), -l["ni"]))
        for lot in candidates:
            if remaining <= 0:
                break
            take = min(lot["avail"], max(x_min, remaining))
            if take > remaining and remaining >= x_min:
                take = remaining
            if take >= x_min or math.isclose(take, remaining):
                x_map[lot["id"]] = take
                remaining -= take
        if remaining != 0 and x_map:
            last = list(x_map.keys())[-1]
            x_map[last] = max(0, x_map[last] + remaining)
        rows_df, kpis = package_solution(lots, x_map, T, G, eps)
        return rows_df, kpis, f"Greedy-Fallback ({type(e).__name__}: {e})"

# --------------- Inventory & leftovers ---------------
def apply_allocation(norm_df, alloc_df):
    if alloc_df is None or alloc_df.empty:
        return norm_df.copy()
    left = norm_df.copy()
    use = alloc_df.groupby("LotID", as_index=False)["WMT_to_load"].sum()
    use["LotID"] = use["LotID"].astype(str)
    left["LotID"] = left["LotID"].astype(str)
    left = left.merge(use, on="LotID", how="left", suffixes=("", "_used"))
    left["WMT_to_load"] = left["WMT_to_load"].fillna(0.0)
    left["WMT_available"] = (left["WMT_available"] - left["WMT_to_load"]).clip(lower=0)
    left = left.drop(columns=["WMT_to_load"])
    return left

def compute_leftover_inventory(norm_df, *allocation_dfs):
    used = pd.DataFrame(columns=["LotID", "WMT_to_load"])
    for df in allocation_dfs:
        if df is None or len(df) == 0:
            continue
        part = df[["LotID", "WMT_to_load"]].copy()
        used = pd.concat([used, part], ignore_index=True)
    used = used.groupby("LotID", as_index=False)["WMT_to_load"].sum()
    inv = norm_df.copy()
    inv["LotID"] = inv["LotID"].astype(str); used["LotID"] = used["LotID"].astype(str)
    inv = inv.merge(used, on="LotID", how="left", suffixes=("", "_used"))
    inv["WMT_to_load"] = inv["WMT_to_load"].fillna(0.0)
    inv["WMT_remaining"] = (inv["WMT_available"] - inv["WMT_to_load"]).clip(lower=0)
    inv["Ni_units_remaining_t"] = inv["WMT_remaining"] * inv["Ni"] / 100.0
    leftover = inv[["LotID", "Ni", "WMT_available", "WMT_to_load", "WMT_remaining", "Ni_units_remaining_t"]].rename(columns={"Ni": "Ni_%"})
    totals = {
        "Total_WMT_remaining": round(float(leftover["WMT_remaining"].sum()), 2),
        "Total_Ni_units_remaining_t": round(float(leftover["Ni_units_remaining_t"].sum()), 4),
    }
    rem = leftover[leftover["WMT_remaining"] > 0]
    totals["Avg_Ni_remaining_%"] = round(float((rem["WMT_remaining"] * rem["Ni_%"]).sum() / rem["WMT_remaining"].sum()), 4) if not rem.empty else None
    return leftover, totals

def to_xlsx_sheets(sheets: dict, filename: str):
    bio = BytesIO()
    with pd.ExcelWriter(bio, engine="openpyxl") as writer:
        for name, df in sheets.items():
            (df if isinstance(df, pd.DataFrame) else pd.DataFrame(df)).to_excel(
                writer, sheet_name=str(name)[:31], index=False
            )
    bio.seek(0)
    st.download_button(
        "Download XLSX",
        data=bio.getvalue(),
        file_name=filename,
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

# --------------------------- UI ---------------------------
st.sidebar.header("Inputs")
uploaded = st.sidebar.file_uploader("Upload PSI CSV/XLSX", type=["csv", "xlsx"])

with st.sidebar.expander("Column mapping (optional)"):
    lot_col_opt = st.text_input("Lot ID column name (optional)", value="")
    wmt_col_opt = st.text_input("Available WMT column name (optional)", value="")
    use_positions = st.checkbox("Use positions A (LotID) and K (WMT)", value=True)

x_min = st.sidebar.number_input("Minimum per used lot (WMT)", min_value=0, max_value=10000, value=300, step=50)
prefer_min_lots = st.sidebar.checkbox("Prefer fewer lots per cargo", value=True)

app_mode = st.sidebar.selectbox("App Mode", ["Multi-Cargo", "Scenario Builder"], index=0)

if uploaded is not None:
    df, delim, header_idx = read_dataframe(uploaded)
    st.success(f"Detected delimiter: '{delim}', header row: {header_idx+1}")
    mapping = map_columns(df, lot_col_opt or None, wmt_col_opt or None)
    pos_map = {"A": 0, "K": 10} if ('use_positions' not in locals() or use_positions) else None
    norm_df = normalize_df(df, mapping, pos_map)
    st.subheader("Normalized Preview")
    st.dataframe(norm_df.head(50))

    if {"LotID", "WMT_available", "Ni"}.issubset(norm_df.columns):
        if app_mode == "Multi-Cargo":
            # ------- Multi-Cargo mode -------
            cargo_wmt = st.sidebar.number_input("Cargo WMT", min_value=0, max_value=200000, value=7500, step=100)
            target_ni = st.sidebar.number_input("Ni floor %", min_value=0.0, max_value=5.0, value=1.43, step=0.01)
            k_mode = st.sidebar.selectbox("How many cargoes?", ["Auto (as many as possible)"] + [str(i) for i in range(1, 51)], index=0)

            def balance_multi_cargoes(norm_df_local, T, floor, x_min_local, prefer_min_lots_local=True, k=None, k_max_cap=50):
                results = []
                inv = norm_df_local.copy()
                if inv["WMT_available"].sum() < T:
                    return results, inv
                count = 0
                while (k is None and count < k_max_cap) or (k is not None and count < k):
                    cur = inv[inv["WMT_available"] > 0].copy()
                    if cur["WMT_available"].sum() < T:
                        break
                    rows_df, kpis, solver = optimize_single_cargo(cur, T, floor, None, x_min_local, prefer_min_lots_local)
                    good = (not rows_df.empty) and abs(rows_df["WMT_to_load"].sum() - T) < 1e-3 and (kpis["Weighted_Ni_%"] is not None and kpis["Weighted_Ni_%"] >= floor - 1e-12)
                    if not good:
                        break
                    results.append((rows_df, kpis, solver))
                    inv = apply_allocation(inv, rows_df)
                    count += 1
                return results, inv

            k = None if k_mode.startswith("Auto") else int(k_mode)
            results, _leftover_inv = balance_multi_cargoes(norm_df, cargo_wmt, target_ni, x_min, prefer_min_lots, k=k)
            if not results:
                st.error("No feasible cargo could be built with current settings (check Ni floor, cargo size, or x_min).")
            else:
                all_sheets = {}
                for i, (df_out, kpis, solver) in enumerate(results, start=1):
                    st.subheader(f"Cargo {i}")
                    st.dataframe(df_out)
                    st.write({
                        "Cargo_WMT": df_out["WMT_to_load"].sum(),
                        "Weighted_Ni_%": kpis["Weighted_Ni_%"],
                        "Lots_Used": kpis["Lots_Used"],
                        "Meets_Ni_floor": (kpis["Weighted_Ni_%"] is not None and kpis["Weighted_Ni_%"] >= target_ni - 1e-12),
                        "Solver": solver
                    })
                    st.download_button(
                        f"Download Cargo {i} CSV",
                        (df_out.to_csv(index=False) if not df_out.empty else "").encode("utf-8"),
                        file_name=f"cargo_{i}_{int(cargo_wmt)}wmt_floor{target_ni}.csv"
                    )
                    all_sheets[f"Cargo {i}"] = df_out

                st.subheader("Leftovers")
                for i in range(len(results)):
                    partial = [r[0] for r in results[:i+1]]
                    lv, tot = compute_leftover_inventory(norm_df, *partial)
                    st.markdown(f"**After Cargo {i+1}**")
                    st.dataframe(lv)
                    st.write(tot)
                    all_sheets[f"Leftover after C{i+1}"] = lv
                    all_sheets[f"Totals after C{i+1}"] = pd.DataFrame([tot])

                final_leftover, final_totals = compute_leftover_inventory(norm_df, *[r[0] for r in results])
                st.markdown("**Final Leftover (after all cargoes)**")
                st.dataframe(final_leftover)
                st.write(final_totals)
                all_sheets["Final Leftover"] = final_leftover
                all_sheets["Final Totals"] = pd.DataFrame([final_totals])

                to_xlsx_sheets(all_sheets, filename=f"multi_cargo_{len(results)}x{int(cargo_wmt)}wmt_floor{target_ni}.xlsx")

        else:
            # ------- Scenario Builder mode -------
            st.subheader("Scenario Builder")
            num_cargo = st.sidebar.number_input("Number of cargoes in scenario", min_value=1, max_value=20, value=3, step=1)

            # Defaults: first cargo target band at 1.30, others Ni floor at 1.43
            default_targets = [1.30] + [1.43]*(num_cargo-1)
            default_modes   = ["Target Band"] + ["Ni Floor"]*(num_cargo-1)
            default_tol     = [0.02] + [0.02]*(num_cargo-1)
            default_wmt     = [7500]*num_cargo

            scenario = []
            for i in range(num_cargo):
                with st.sidebar.expander(f"Cargo {i+1} settings", expanded=(i==0)):
                    wmt = st.number_input(f"Cargo {i+1} WMT", min_value=0, max_value=200000, value=int(default_wmt[i]), step=100, key=f"sb_wmt_{i}")
                    mode = st.selectbox(f"Ni Rule (Cargo {i+1})", ["Ni Floor", "Target Band"], index=0 if default_modes[i]=="Ni Floor" else 1, key=f"sb_mode_{i}")
                    target = st.number_input(f"Ni % (Cargo {i+1})", min_value=0.0, max_value=5.0, value=float(default_targets[i]), step=0.01, key=f"sb_target_{i}")
                    tol = None
                    if mode == "Target Band":
                        tol = st.number_input(f"Tolerance ±% (Cargo {i+1})", min_value=0.0, max_value=1.0, value=float(default_tol[i]), step=0.005, key=f"sb_tol_{i}")
                    scenario.append({"wmt": wmt, "mode": mode, "target": target, "tol": tol})

            inv = norm_df.copy()
            cargo_allocs = []
            sheets = {}

            for i, cfg in enumerate(scenario, start=1):
                T = cfg["wmt"]
                if inv["WMT_available"].sum() < T:
                    st.warning(f"Cargo {i}: Not enough remaining WMT to fill {T}. Stopping here.")
                    break
                eps = None if cfg["mode"] == "Ni Floor" else cfg["tol"]
                rows_df, kpis, solver = optimize_single_cargo(inv, T, cfg["target"], eps, x_min, prefer_min_lots)

                total = rows_df["WMT_to_load"].sum() if not rows_df.empty else 0.0
                avg = kpis.get("Weighted_Ni_%")
                feasible = (
                    (abs(total - T) < 1e-3) and
                    (avg is not None) and
                    ((eps is None and avg >= cfg["target"] - 1e-12) or
                     (eps is not None and (cfg["target"]-eps-1e-12) <= avg <= (cfg["target"]+eps+1e-12)))
                )

                st.subheader(f"[Scenario] Cargo {i} — {'OK' if feasible else 'FAILED'}")
                st.dataframe(rows_df)
                st.write({
                    "Cargo_WMT": total,
                    "Weighted_Ni_%": avg,
                    "Lots_Used": kpis.get("Lots_Used"),
                    "Rule": cfg["mode"],
                    "Target": cfg["target"],
                    "Tol": eps,
                    "Solver": solver
                })

                if not feasible:
                    st.error(f"Cargo {i} couldn't meet the rule. Stopping the scenario here.")
                    break

                cargo_allocs.append(rows_df)
                sheets[f"Cargo {i}"] = rows_df
                inv = apply_allocation(inv, rows_df)

                lv, tot = compute_leftover_inventory(norm_df, *cargo_allocs)
                st.markdown(f"**[Scenario] Leftover AFTER Cargo {i}**")
                st.dataframe(lv)
                st.write(tot)
                sheets[f"Leftover after C{i}"] = lv
                sheets[f"Totals after C{i}"] = pd.DataFrame([tot])

            if cargo_allocs:
                final_leftover, final_totals = compute_leftover_inventory(norm_df, *cargo_allocs)
                st.subheader("[Scenario] Final Leftover")
                st.dataframe(final_leftover)
                st.write(final_totals)
                sheets["Final Leftover"] = final_leftover
                sheets["Final Totals"] = pd.DataFrame([final_totals])
                to_xlsx_sheets(sheets, filename=f"scenario_{len(cargo_allocs)}_cargoes.xlsx")
            else:
                st.info("Scenario built 0 cargo. Adjust Ni rules, WMT, or min-per-lot settings.")
    else:
        st.error("Could not find required columns: LotID, WMT_available, Ni. Use the mapping inputs or fix the header row.")
else:
    st.info("Upload a PSI CSV/XLSX to begin.")
