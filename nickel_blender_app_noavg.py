
import io
import math
import pandas as pd
import numpy as np
import streamlit as st

# No-average policy:
#  - We DO NOT minimize |avgNi - target| anymore.
#  - We ONLY enforce Ni constraints:
#       * For "target ± tolerance" mode: (G-eps)*T <= sum(x_i * Ni_i) <= (G+eps)*T
#       * For "Ni ≥ floor" modes:        sum(x_i * Ni_i) >= G*T
#  - Objective: MINIMIZE number of lots used (simpler operations) with x_min per used lot.

def detect_header_and_delim(file_bytes):
    text = file_bytes.decode("utf-8", errors="ignore")
    lines = text.splitlines()
    delim = ";" if lines and lines[0].count(";") >= 3 else ","
    header_idx = 0
    tokens = ["lot","wmt","wet","ni","fe","sio2","mgo","moist"]
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
    ni_col  = find_col(["ni(%)","ni %","ni","nickel","ni grade"])
    fe_col  = find_col(["fe(%)","fe %","fe"])
    sio2_col= find_col(["sio2(%)","sio2 %","sio2"])
    mgo_col = find_col(["mgo(%)","mgo %","mgo"])
    moist_col = find_col(["moisture(%)","moisture %","moisture","h2o","h2o%"])
    mapping = {"LotID": lot_col, "WMT_available": wmt_col, "Ni": ni_col,
               "Fe": fe_col, "SiO2": sio2_col, "MgO": mgo_col, "Moisture": moist_col}
    return mapping

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
    for c in ["WMT_available","Ni","Fe","SiO2","MgO","Moisture"]:
        if c in norm.columns:
            norm[c] = pd.to_numeric(norm[c].astype(str).str.replace(",", ".", regex=False)
                                    .str.replace(r"[^0-9\.\-]", "", regex=True), errors="coerce")
    core = [c for c in ["LotID","WMT_available","Ni"] if c in norm.columns]
    if core:
        norm = norm.dropna(subset=core, how="any")
    if "WMT_available" in norm.columns:
        norm = norm[norm["WMT_available"] > 0]
    return norm

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
                "Ni_%": lot["ni"]
            })
            total_wmt += x
            numer += x * lot["ni"]
    rows_df = pd.DataFrame(rows).sort_values(by="WMT_to_load", ascending=False)
    avg_ni = numer / total_wmt if total_wmt > 0 else float("nan")
    lots_used = (rows_df["WMT_to_load"] > 0).sum() if len(rows_df) else 0
    kpis = {
        "Total_WMT": round(total_wmt,2),
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
        # Objective: minimize lots used
        prob.setObjective(pl.lpSum(y[lot["id"]]) if prefer_min_lots else 0)
        prob.solve(pl.PULP_CBC_CMD(msg=False))
        x_map = {lot["id"]: float(pl.value(x[lot["id"]]) or 0) for lot in lots}
        rows_df, kpis = package_solution(lots, x_map, T, G, eps)
        return rows_df, kpis, "PuLP-CBC"
    except Exception as e:
        remaining = T
        x_map = {}
        if eps is None:
            candidates = sorted([l for l in lots if l["avail"] >= x_min], key=lambda l: (-1 if l["ni"]>=G else 1, -l["ni"]))
        else:
            candidates = sorted([l for l in lots if l["avail"] >= x_min], key=lambda l: (abs(l["ni"]-G), -l["ni"]))
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

def max_cargo_ge_floor(norm_df, floor, x_min):
    df = norm_df.copy()
    df = df[df["WMT_available"] >= x_min]
    df["delta"] = df["Ni"] - floor
    high = df[df["delta"] >= 0].copy()
    low  = df[df["delta"] < 0].copy().sort_values(by="Ni", ascending=False)
    selected = []
    total = 0.0
    surplus = 0.0
    for _, r in high.iterrows():
        x = float(r["WMT_available"])
        total += x
        surplus += x * (r["Ni"] - floor)
        selected.append({"LotID": r["LotID"], "Ni_%": r["Ni"], "WMT_to_load": x, "Reason": "Ni>=floor (full)"})
    for _, r in low.iterrows():
        deficit = floor - r["Ni"]
        if deficit <= 0: continue
        cap = surplus / deficit if surplus > 0 else 0.0
        if cap <= 0: continue
        take = min(float(r["WMT_available"]), cap)
        if take >= x_min + 1e-9:
            total += take
            surplus -= take * deficit
            selected.append({"LotID": r["LotID"], "Ni_%": r["Ni"], "WMT_to_load": float(take), "Reason": "Ni<floor (capped)"})
    sel = pd.DataFrame(selected)
    if sel.empty:
        return sel, {"Max_Cargo_WMT": 0, "Weighted_Ni_%": None, "Lots_Used": 0}
    avg = (sel["WMT_to_load"] * sel["Ni_%"]).sum() / sel["WMT_to_load"].sum()
    return sel, {"Max_Cargo_WMT": round(sel['WMT_to_load'].sum(),2), "Weighted_Ni_%": round(avg,4), "Lots_Used": int(sel.shape[0])}

def balance_two_cargoes(norm_df, T, floor, x_min):
    inv = {str(r["LotID"]): {"avail": float(r["WMT_available"]), "ni": float(r["Ni"])} for _, r in norm_df.iterrows()}
    cargoes = [{"remaining": T, "alloc": {}, "surplus": 0.0},
               {"remaining": T, "alloc": {}, "surplus": 0.0}]
    cand = [(k, v["ni"], v["avail"]) for k,v in inv.items() if v["avail"] >= x_min]
    high = [x for x in cand if x[1] >= floor]; high.sort(key=lambda t: t[1], reverse=True)
    low  = [x for x in cand if x[1] < floor];  low.sort(key=lambda t: t[1], reverse=True)
    for lot, ni, avail in high:
        rem = avail
        while rem >= x_min:
            idx = 0 if cargoes[0]["surplus"] <= cargoes[1]["surplus"] else 1
            cg = cargoes[idx]
            if cg["remaining"] <= 0: idx = 1-idx; cg = cargoes[idx]
            if cg["remaining"] <= 0: break
            take = min(rem, cg["remaining"])
            if take < x_min and cg["remaining"] >= x_min:
                take = min(x_min, rem)
            if take < x_min and not math.isclose(take, cg["remaining"]):
                break
            cg["alloc"][lot] = cg["alloc"].get(lot, 0.0) + take
            cg["remaining"] -= take
            cg["surplus"] += take * (ni - floor)
            rem -= take
            if cargoes[0]["remaining"] <= 0 and cargoes[1]["remaining"] <= 0:
                break
    for lot, ni, avail in low:
        rem = avail
        while rem >= x_min:
            idx = 0 if cargoes[0]["surplus"] >= cargoes[1]["surplus"] else 1
            if cargoes[idx]["remaining"] <= 0: idx = 1-idx
            if cargoes[idx]["remaining"] <= 0: break
            cg = cargoes[idx]
            deficit = (floor - ni)
            cap = cg["surplus"] / deficit if deficit > 0 else cg["remaining"]
            if cap <= 0: break
            take = min(rem, cg["remaining"], cap)
            if take < x_min and cg["remaining"] >= x_min:
                if x_min <= min(rem, cap) + 1e-9:
                    take = x_min
                else:
                    break
            if take < x_min and not math.isclose(take, cg["remaining"]):
                break
            cg["alloc"][lot] = cg["alloc"].get(lot, 0.0) + take
            cg["remaining"] -= take
            cg["surplus"] -= take * (floor - ni)
            rem -= take
            if cargoes[0]["remaining"] <= 0 and cargoes[1]["remaining"] <= 0:
                break
    outs = []
    for cg in cargoes:
        df_out = pd.DataFrame([{"LotID": k, "Ni_%": inv[k]["ni"], "WMT_to_load": v} for k,v in cg["alloc"].items()])
        if df_out.empty:
            outs.append((df_out, None, False)); continue
        total = df_out["WMT_to_load"].sum()
        avg = (df_out["WMT_to_load"] * df_out["Ni_%"]).sum() / total if total>0 else None
        meets = (abs(total - T) < 1e-6) and (avg is not None and avg >= floor - 1e-12)
        outs.append((df_out, avg, meets))
    return outs

# =============================
# UI
# =============================
st.set_page_config(page_title="Nickel Blending Optimizer (No-Average Objective)", layout="wide")
st.title("🛠️ Nickel Blending Optimizer — No Average in Objective")

st.sidebar.header("Inputs")
uploaded = st.sidebar.file_uploader("Upload PSI CSV/XLSX", type=["csv","xlsx"])
lot_col_opt = ""
wmt_col_opt = ""
with st.sidebar.expander("Column mapping (optional)"):
    st.write("If your file doesn't have clear headers, you can force mapping:")
    lot_col_opt = st.text_input("Lot ID column name (optional)", value="")
    wmt_col_opt = st.text_input("Available WMT column name (optional)", value="")
    use_positions = st.checkbox("Use positions A (LotID) and K (WMT)", value=True)

cargo_wmt = st.sidebar.number_input("Cargo WMT", min_value=0, max_value=20000, value=7500, step=100)
mode = st.sidebar.selectbox("Mode", [
    "Single Cargo (target band only — no centering)",
    "Single Cargo (Ni ≥ floor)",
    "Max Cargo (Ni ≥ floor)",
    "Balanced Two Cargoes (Ni ≥ floor)"
], index=3)
target_ni = st.sidebar.number_input("Target Ni % / Floor", min_value=0.0, max_value=5.0, value=1.43, step=0.01)
tolerance = st.sidebar.number_input("Tolerance ±%", min_value=0.0, max_value=1.0, value=0.02, step=0.005, help="Used only in the target band mode; we do not minimize deviation.")
x_min = st.sidebar.number_input("Minimum per used lot (WMT)", min_value=0, max_value=5000, value=300, step=50)
prefer_min_lots = st.sidebar.checkbox("Prefer fewer lots (objective)", value=True)

if uploaded is not None:
    try:
        df, delim, header_idx = read_dataframe(uploaded)
        st.success(f"Detected delimiter: '{delim}', header row: {header_idx+1}")
        mapping = map_columns(df, lot_col_opt or None, wmt_col_opt or None)
        pos_map = {"A":0, "K":10} if ('use_positions' not in locals() or use_positions) else None
        norm_df = normalize_df(df, mapping, pos_map)
        st.subheader("Normalized Preview")
        st.dataframe(norm_df.head(50))
        if {"LotID","WMT_available","Ni"}.issubset(norm_df.columns):
            if mode == "Single Cargo (target band only — no centering)":
                rows_df, kpis, solver = optimize_single_cargo(norm_df, cargo_wmt, target_ni, tolerance, x_min, prefer_min_lots)
                st.subheader("Blend (Target Band, No Centering)")
                st.dataframe(rows_df); st.write(kpis); st.caption(f"Solver: {solver}")
                st.download_button("Download CSV", rows_df.to_csv(index=False), file_name=f"blend_{cargo_wmt}wmt_band{target_ni}_noavg.csv")
            elif mode == "Single Cargo (Ni ≥ floor)":
                rows_df, kpis, solver = optimize_single_cargo(norm_df, cargo_wmt, target_ni, None, x_min, prefer_min_lots)
                st.subheader("Blend (Ni ≥ Floor)")
                st.dataframe(rows_df); st.write(kpis); st.caption(f"Solver: {solver}")
                st.download_button("Download CSV", rows_df.to_csv(index=False), file_name=f"blend_{cargo_wmt}wmt_floor{target_ni}_noavg.csv")
            elif mode == "Max Cargo (Ni ≥ floor)":
                sel, summary = max_cargo_ge_floor(norm_df, target_ni, x_min)
                st.subheader("Max Cargo (Ni ≥ Floor)")
                st.dataframe(sel); st.write(summary)
                st.download_button("Download CSV", sel.to_csv(index=False), file_name=f"maxcargo_floor{target_ni}_noavg.csv")
            else:
                outs = balance_two_cargoes(norm_df, cargo_wmt, target_ni, x_min)
                for i, (df_out, avg, meets) in enumerate(outs, start=1):
                    st.subheader(f"Cargo {i} (Balanced)")
                    st.dataframe(df_out)
                    st.write({"Cargo_WMT": df_out["WMT_to_load"].sum() if not df_out.empty else 0,
                              "Weighted_Ni_%": round(avg,4) if avg is not None else None,
                              "Meets_Ni_floor": bool(meets)})
                    st.download_button(f"Download Cargo {i} CSV", (df_out.to_csv(index=False) if not df_out.empty else "").encode("utf-8"),
                                       file_name=f"balanced_cargo_{i}_{cargo_wmt}wmt_floor{target_ni}_noavg.csv")
        else:
            st.error("Could not find required columns: LotID, WMT_available, Ni. Use the mapping inputs or fix the header row.")
    except Exception as e:
        st.exception(e)
else:
    st.info("Upload a PSI CSV/XLSX to begin.")
