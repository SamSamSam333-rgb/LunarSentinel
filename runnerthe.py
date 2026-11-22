import folium
from folium.plugins import HeatMap
import pandas as pd
import numpy as np
import json
import os
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests as req

# ── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__, static_folder='.')
CORS(app)

# Gemini API key: set the GEMINI_API_KEY environment variable before running.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

@app.route('/gemini', methods=['POST'])
def gemini_proxy():
    try:
        data = request.json
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
        response = req.post(url, json=data, headers={'Content-Type': 'application/json'})
        result = response.json()
        print("Gemini status:", response.status_code)
        if response.status_code != 200:
            print("Gemini error:", result)
        return jsonify(result), response.status_code
    except Exception as e:
        print("Proxy error:", str(e))
        return jsonify({"error": str(e)}), 500

# ── Real-time causes & solutions (Gemini, seeded with real computed metrics) ─
causes_solutions_cache = {}

def build_causes_solutions_prompt(loc, d):
    return f"""You are an environmental scientist analysing REAL Sentinel-2-derived water quality
data for {d['display']}, Maharashtra, India. These numbers come from an actual pixel classifier
trained on real satellite reflectance and calibrated against real MPCB pollution reports — they
are not invented.

Real computed data for {d['display']}:
- 2021-2025 average: {d['clean_pct']}% clean, {d['turb_pct']}% turbid, {d['poll_pct']}% polluted
- Risk grade: {d['grade']} (risk score {d['risk']})
- Trend 2021->2025 polluted%: {d['trend_series']}
- Forecast 2026-2028 polluted% (model: {d['best_model']}): {d['forecast_avg']}
- Calibrated spectral thresholds used for classification: NDWI_polluted < {d['ndwi_thr']}, Turbidity_polluted > {d['turb_thr']}

Based on this REAL data and your general knowledge of Maharashtra coastal environmental issues
(ports, rivers, industry, tourism, agriculture, sewage infrastructure in this specific area),
return STRICT JSON only, no markdown, in this exact shape:
{{
  "causes": [
    {{"name": "short cause name with one emoji prefix", "pct": <int 5-45>, "desc": "1-2 sentence real-world explanation tied to the actual trend shown above"}}
  ],
  "solutions": [
    {{"priority": "URGENT|SHORT-TERM|LONG-TERM|POLICY", "title": "short title", "desc": "1-2 sentence concrete, location-specific recommendation"}}
  ]
}}
Give 4-5 causes (pct values should roughly sum to 100) and 4-5 solutions. Be specific to
{d['display']} and consistent with the real trend direction shown above (rising, falling, or
stable). Do not invent statistics that contradict the real numbers given."""

def call_gemini_json(prompt):
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={GEMINI_API_KEY}"
    body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}]}
    resp = req.post(url, json=body, headers={'Content-Type': 'application/json'}, timeout=30)
    resp.raise_for_status()
    text = resp.json()['candidates'][0]['content']['parts'][0]['text']
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    return json.loads(text)

@app.route('/causes_solutions/<loc>', methods=['GET'])
def causes_solutions(loc):
    refresh = request.args.get('refresh', '0') == '1'
    if loc not in intel:
        return jsonify({"error": "unknown location"}), 404
    if not refresh and loc in causes_solutions_cache:
        return jsonify(causes_solutions_cache[loc])
    try:
        prompt = build_causes_solutions_prompt(loc, intel[loc])
        result = call_gemini_json(prompt)
        causes_solutions_cache[loc] = result
        return jsonify(result)
    except Exception as e:
        print("causes_solutions error:", e)
        fallback = {"causes": [], "solutions": [],
                    "error": f"Live generation failed ({e}). This section is real-time Gemini output seeded with the real metrics above — retry with the refresh button."}
        return jsonify(fallback), 200

HTML_PATH = BASE_DIR

@app.route('/')
def serve_dashboard():
    return send_from_directory(HTML_PATH, 'ai_water_quality_dashboard.html')

# ── Load REAL computed data (from the verified pipeline — no synthetic values) ──
yearly = pd.read_csv(os.path.join(DATA_DIR, "yearly_stats.csv"))
monthly_full = pd.read_csv(os.path.join(DATA_DIR, "monthly_full_stats.csv"))
forecast_df = pd.read_csv(os.path.join(DATA_DIR, "forecast_2026_2028.csv"))
model_comp = pd.read_csv(os.path.join(DATA_DIR, "forecast_model_comparison.csv"))
classifier_metrics = pd.read_csv(os.path.join(DATA_DIR, "classifier_metrics.csv"))
confusion = pd.read_csv(os.path.join(DATA_DIR, "confusion_matrix_ensemble.csv"), index_col=0)
coverage = pd.read_csv(os.path.join(DATA_DIR, "coverage_matrix.csv"))
thresholds = pd.read_csv(os.path.join(DATA_DIR, "calibrated_thresholds_per_location.csv")).set_index("location")

coords = {
    "mumbai":     (18.9333, 72.8777),
    "navi_mumbai":(19.0330, 73.0297),
    "alibag":     (18.6500, 72.8777),
    "ratnagiri":  (16.9902, 73.3000),
}
display_names = {"mumbai":"Mumbai","navi_mumbai":"Navi Mumbai","alibag":"Alibag","ratnagiri":"Ratnagiri"}
color_map    = {"clean": "#2E86DE", "polluted": "#EE5253", "turbid": "#10AC84"}
grade_colors = {"A": "#10AC84", "B": "#8BC34A", "C": "#F0932B", "D": "#EE5253"}
years = [2021,2022,2023,2024,2025]
months_used = [1,2,3,4,5,10,11,12]

def get_risk_score(poll, turb):
    score = round(poll*1.0 + turb*0.5, 1)
    if score < 10: grade = "A"
    elif score < 25: grade = "B"
    elif score < 50: grade = "C"
    else: grade = "D"
    return score, grade

def year_stats_for(loc):
    sub = yearly[yearly.location==loc].set_index("year")
    return {yr: dict(clean=round(sub.loc[yr,'clean_pct'],1), turbid=round(sub.loc[yr,'turbid_pct'],1),
                      polluted=round(sub.loc[yr,'polluted_pct'],1)) for yr in years}

def forecast_for(loc):
    sub = forecast_df[forecast_df.location==loc]
    out = {}
    for yr in [2026,2027,2028]:
        ysub = sub[sub.year==yr]
        out[yr] = dict(polluted=round(ysub.forecast_polluted_pct.mean(),1))
    model_used = sub['model_used'].iloc[0] if len(sub) else "n/a"
    return out, model_used

overall = {}
year_stats = {}
forecasts = {}
best_models = {}
for loc in coords:
    ys = year_stats_for(loc); year_stats[loc] = ys
    fc, model_used = forecast_for(loc); forecasts[loc] = fc; best_models[loc] = model_used
    c = np.mean([ys[y]['clean'] for y in years]); t = np.mean([ys[y]['turbid'] for y in years]); p = np.mean([ys[y]['polluted'] for y in years])
    rs, gr = get_risk_score(p, t)
    trend = "worsening" if ys[2025]['polluted'] > ys[2021]['polluted']+3 else ("improving" if ys[2025]['polluted'] < ys[2021]['polluted']-3 else "stable")
    overall[loc] = dict(clean=round(c,1), polluted=round(p,1), turbid=round(t,1), risk=rs, grade=gr, trend=trend)

model_meta = {}
for loc in coords:
    row = model_comp[(model_comp.location==loc) & (model_comp.is_best==True)].iloc[0]
    model_meta[loc] = dict(model=row['model'], rmse=row['RMSE'], mae=row['MAE'], r2=row['R2'], within5=row['Within5pp'])

intel = {}
for loc in coords:
    o = overall[loc]
    thr = thresholds.loc[loc]
    trend_series = [year_stats[loc][y]['polluted'] for y in years]
    fc_avg = [forecasts[loc][y]['polluted'] for y in [2026,2027,2028]]
    intel[loc] = dict(
        display=display_names[loc], grade=o['grade'], risk=o['risk'],
        clean_pct=o['clean'], poll_pct=o['polluted'], turb_pct=o['turbid'], trend=o['trend'],
        dominant="polluted" if o['polluted']>o['clean'] and o['polluted']>o['turbid'] else ("turbid" if o['turbid']>o['clean'] else "clean"),
        best_model=model_meta[loc]['model'], model_rmse=model_meta[loc]['rmse'],
        trend_series=trend_series, forecast_avg=fc_avg,
        ndwi_thr=thr['ndwi_polluted_threshold'], turb_thr=thr['turbidity_polluted_threshold'],
    )

ai_context = f"""You are an expert environmental scientist analysing REAL Sentinel-2 satellite
data for Maharashtra's coast (Mumbai, Navi Mumbai, Alibag, Ratnagiri), 2021-2025, real pixels
classified by a trained RF+GradientBoosting+DNN ensemble (majority vote), thresholds calibrated
against real MPCB pollution reports.

REAL 2021-2025 average water quality (clean / turbid / polluted %, risk grade):
""" + "\n".join([f"- {display_names[l]}: {overall[l]['clean']}% clean, {overall[l]['turbid']}% turbid, {overall[l]['polluted']}% polluted | Grade {overall[l]['grade']} | Risk {overall[l]['risk']} | Trend: {overall[l]['trend']}" for l in coords]) + f"""

REAL classifier performance (held-out 20% test set, majority-vote ensemble):
accuracy {classifier_metrics[classifier_metrics.model=='Ensemble_MajorityVote'].accuracy.iloc[0]*100:.1f}%,
macro F1 {classifier_metrics[classifier_metrics.model=='Ensemble_MajorityVote'].f1_macro.iloc[0]:.3f}.
Known limitation: POLLUTED-class recall is weaker (~0.54) because POLLUTED and TURBID sit either
side of a calibrated spectral threshold rather than being sharply distinct — always mention this
if asked about classifier accuracy.

REAL forecast 2026-2028 (best model chosen per location by lowest test RMSE on real 2025 holdout):
""" + "\n".join([f"- {display_names[l]}: {model_meta[l]['model']} (test RMSE={model_meta[l]['rmse']}), forecast avg polluted% 2026/27/28 = {intel[l]['forecast_avg']}" for l in coords]) + """

Answer questions specifically and honestly using these real numbers. If asked about accuracy,
mention the real ensemble metrics above, not an inflated figure. If asked why a number differs
from expectation, refer to the calibration/limitation notes. Keep answers concise, scientific,
and grounded only in the data given here."""

ai_context_json = json.dumps(ai_context)

# ── Folium map ────────────────────────────────────────────────────────────────
m = folium.Map(location=[18.2, 73.0], zoom_start=8, tiles="CartoDB positron")
folium.TileLayer("CartoDB dark_matter", name="Dark Mode").add_to(m)

current_group = folium.FeatureGroup(name="Water Quality", show=True)
heatmap_group = folium.FeatureGroup(name="Pollution Heatmap", show=True)
year_groups = {yr: folium.FeatureGroup(name=f"Year {yr}", show=False) for yr in years}
forecast_groups = {yr: folium.FeatureGroup(name=f"Forecast {yr}", show=False) for yr in [2026,2027,2028]}

def mini_bar(s, label_map=("clean","polluted","turbid")):
    parts = ""
    for key in label_map:
        val = s.get(key, 0)
        parts += f"<div style='width:{max(val,1.5)}%;background:{color_map[key]}'></div>"
    return f"<div style='display:flex;height:10px;border-radius:5px;overflow:hidden;background:#eee'>{parts}</div>"

for loc,(lat,lon) in coords.items():
    d = intel[loc]; o = overall[loc]; gc = grade_colors[d['grade']]; dom_color = color_map[d['dominant']]

    rows_html = ""
    for yr in years:
        s = year_stats[loc][yr]
        rows_html += f"""<div style='margin:4px 0;font-size:11px'>
          <div style='display:flex;justify-content:space-between'><b>{yr}</b><span>C:{s['clean']}% T:{s['turbid']}% P:{s['polluted']}%</span></div>
          {mini_bar(s)}</div>"""
    for yr in [2026,2027,2028]:
        p = forecasts[loc][yr]['polluted']
        rows_html += f"""<div style='margin:4px 0;font-size:11px;border-left:2px dashed #999;padding-left:6px'>
          <div style='display:flex;justify-content:space-between'><b>{yr} (forecast)</b><span>P:{p}%</span></div>
          <div style='font-size:9px;color:#666'>model: {best_models[loc]}</div></div>"""

    popup_html = f"""
    <div style='font-family:Segoe UI,Arial;width:280px'>
      <div style='background:{dom_color};color:white;padding:10px 12px;border-radius:8px 8px 0 0'>
        <b style='font-size:15px'>{d['display']}</b><br>
        <span style='font-size:11px;opacity:0.9'>Grade {d['grade']} · Risk {d['risk']}</span>
      </div>
      <div style='padding:10px 12px;background:white;border-radius:0 0 8px 8px;box-shadow:0 2px 8px rgba(0,0,0,0.15)'>
        {rows_html}
      </div>
    </div>"""

    folium.Circle(location=[lat,lon], radius=9000, color=dom_color, fill=True, fill_color=dom_color,
                  fill_opacity=0.28, weight=2, popup=folium.Popup(popup_html, max_width=300),
                  tooltip=f"{d['display']} — Grade {d['grade']}").add_to(current_group)
    folium.CircleMarker(location=[lat,lon], radius=9, color="white", fill=True, fill_color=dom_color, fill_opacity=1, weight=2).add_to(current_group)
    folium.Marker(location=[lat+0.09, lon+0.02], icon=folium.DivIcon(html=f"""
        <div style='font-size:12px;font-weight:600;color:white;background:{dom_color};padding:4px 10px;
        border-radius:12px;box-shadow:0 2px 6px rgba(0,0,0,0.3);white-space:nowrap;border:2px solid white'>
        {d['display']} · {d['grade']}</div>""")).add_to(current_group)

    for yr in years:
        s = year_stats[loc][yr]
        dom = max(s, key=lambda k: s[k])
        yc = color_map[dom]
        popup_yr = f"""<div style='font-family:Segoe UI;width:200px'>
          <b>{d['display']} — {yr}</b><br>{mini_bar(s)}
          <div style='font-size:11px;margin-top:4px'>Clean {s['clean']}% · Turbid {s['turbid']}% · Polluted {s['polluted']}%</div></div>"""
        folium.Circle(location=[lat,lon], radius=7000, color=yc, fill=True, fill_color=yc, fill_opacity=0.3,
                      popup=folium.Popup(popup_yr, max_width=220), tooltip=f"{d['display']} {yr}").add_to(year_groups[yr])

    for yr in [2026,2027,2028]:
        p = forecasts[loc][yr]['polluted']
        yc = color_map['polluted'] if p > o['polluted'] else color_map['clean']
        popup_fc = f"""<div style='font-family:Segoe UI;width:220px'>
          <b>{d['display']} — {yr} forecast</b><br>Polluted: {p}%<br>
          <span style='font-size:10px;color:#666'>Model: {best_models[loc]} (test RMSE {model_meta[loc]['rmse']})</span></div>"""
        folium.Circle(location=[lat,lon], radius=9000, color=yc, fill=True, fill_color=yc, fill_opacity=0.22,
                      weight=2, dash_array="8", popup=folium.Popup(popup_fc, max_width=240),
                      tooltip=f"{d['display']} {yr} forecast").add_to(forecast_groups[yr])

heat_data = [[coords[loc][0], coords[loc][1], overall[loc]['polluted']/100] for loc in coords]
HeatMap(heat_data, min_opacity=0.35, radius=90, blur=55,
        gradient={"0.2":"#2E86DE","0.4":"#10AC84","0.6":"#F0932B","0.8":"#EE5253","1.0":"#8B0000"}).add_to(heatmap_group)

current_group.add_to(m); heatmap_group.add_to(m)
for yr in years: year_groups[yr].add_to(m)
for yr in [2026,2027,2028]: forecast_groups[yr].add_to(m)
folium.LayerControl(collapsed=False).add_to(m)
map_html = m._repr_html_()

# ── Right-panel data blobs for JS (Chart.js + tables) ─────────────────────────
timeline_chart_data = {}
for loc in coords:
    labels = [str(y) for y in years] + [f"{y}f" for y in [2026,2027,2028]]
    polluted_series = [year_stats[loc][y]['polluted'] for y in years] + [forecasts[loc][y]['polluted'] for y in [2026,2027,2028]]
    clean_series = [year_stats[loc][y]['clean'] for y in years] + [None,None,None]
    timeline_chart_data[loc] = dict(labels=labels, polluted=polluted_series, clean=clean_series)

model_comp_json = model_comp.to_dict(orient='records')
classifier_metrics_json = classifier_metrics.to_dict(orient='records')
confusion_json = dict(labels=list(confusion.columns), rows=[{"label": idx, "values": row.tolist()} for idx, row in confusion.iterrows()])
coverage_json = coverage.to_dict(orient='records')

def make_loc_chips(onclick_name):
    parts = []
    for i, l in enumerate(coords):
        active_cls = " active" if i == 0 else ""
        parts.append(f'<div class="loc-chip{active_cls}" data-loc="{l}" onclick="{onclick_name}(\'{l}\')">{display_names[l]}</div>')
    return "".join(parts)

location_rows_html = ""
for loc in coords:
    d = intel[loc]; c = color_map[d['dominant']]; gc = grade_colors[d['grade']]
    location_rows_html += f"""
    <div class="location-row">
      <span class="loc-name" style="color:{c}">{d['display']}</span>
      <div class="loc-bar-wrap"><div class="loc-bar"><div class="loc-bar-fill" style="width:{d['poll_pct']}%;background:{c}"></div></div>
        <div style="font-size:9px;color:var(--muted);margin-top:2px">Pollution: {d['poll_pct']}%</div></div>
      <span class="loc-grade" style="color:{gc}">{d['grade']}</span>
    </div>"""

methodology_html = f"""
<div class="stat-card">
  <div class="stat-card-title">Ensemble classifier — real held-out test metrics</div>
  <table class="data-table"><thead><tr><th>Model</th><th>Accuracy</th><th>F1 (macro)</th><th>Precision</th><th>Recall</th></tr></thead><tbody>
  {''.join(f"<tr{' class=best' if r['model']=='Ensemble_MajorityVote' else ''}><td>{r['model']}</td><td>{r['accuracy']*100:.1f}%</td><td>{r['f1_macro']:.3f}</td><td>{r['precision_macro']:.3f}</td><td>{r['recall_macro']:.3f}</td></tr>" for r in classifier_metrics_json)}
  </tbody></table>
  <div class="note">Majority vote of Random Forest + Gradient Boosting + DNN (MLP), stratified 80/20 split, evaluated on genuinely held-out pixels.</div>
</div>
<div class="stat-card">
  <div class="stat-card-title">Confusion matrix (ensemble, test set)</div>
  <table class="data-table"><thead><tr><th></th>{''.join(f'<th>pred {l.replace("pred_","")}</th>' for l in confusion_json['labels'])}</tr></thead><tbody>
  {''.join('<tr><td><b>'+r['label'].replace('true_','')+'</b></td>'+''.join(f'<td>{v}</td>' for v in r['values'])+'</tr>' for r in confusion_json['rows'])}
  </tbody></table>
  <div class="note">POLLUTED recall is the known weak point (~0.54) — POLLUTED and TURBID sit either side of a calibrated threshold, not a sharp physical boundary.</div>
</div>
<div class="stat-card">
  <div class="stat-card-title">Forecast model comparison (test RMSE on real 2025 holdout)</div>
  <table class="data-table"><thead><tr><th>Location</th><th>Model</th><th>RMSE</th><th>MAE</th><th>R2</th><th>Within 5pp</th></tr></thead><tbody>
  {''.join(f"<tr{' class=best' if r['is_best'] else ''}><td>{display_names[r['location']]}</td><td>{r['model']}</td><td>{r['RMSE']}</td><td>{r['MAE']}</td><td>{r['R2']}</td><td>{r['Within5pp']}%</td></tr>" for r in model_comp_json)}
  </tbody></table>
  <div class="note">Bold rows = model selected per location by lowest test RMSE.</div>
</div>
<div class="stat-card">
  <div class="stat-card-title">Calibrated classification thresholds (per location)</div>
  <table class="data-table"><thead><tr><th>Location</th><th>NDWI &lt; (polluted)</th><th>Turbidity &gt; (polluted)</th></tr></thead><tbody>
  {''.join(f"<tr><td>{display_names[loc]}</td><td>{thresholds.loc[loc,'ndwi_polluted_threshold']}</td><td>{thresholds.loc[loc,'turbidity_polluted_threshold']}</td></tr>" for loc in coords)}
  </tbody></table>
  <div class="note">NDWI here = (Green-Red)/(Green+Red) — adapted from McFeeters (1996) since no NIR band was exported. Turbidity Index = Red/Blue, a simplified proxy inspired by Nechad et al. (2010).</div>
</div>"""

data_quality_html = f"""
<div class="stat-card">
  <div class="stat-card-title">Real data coverage (pixels per location per month)</div>
  <div class="note" style="margin-bottom:8px">Non-monsoon months (Jan-May, Oct-Dec) are 100% complete across all years. Monsoon months (Jun-Sep) are genuinely inconsistent — not a clean pass/fail.</div>
  <div style="overflow-x:auto"><table class="data-table small"><thead><tr><th>Loc</th><th>Yr</th>{''.join(f'<th>{m:02d}</th>' for m in range(1,13))}</tr></thead><tbody>
  {''.join('<tr><td>'+display_names[r['location']]+'</td><td>'+str(r['year'])+'</td>'+''.join(f"<td class='{'hi' if r[f'month_{m:02d}']>=250 else ('lo' if r[f'month_{m:02d}']>0 else 'zero')}'>{int(r[f'month_{m:02d}'])}</td>" for m in range(1,13))+'</tr>' for r in coverage_json)}
  </tbody></table></div>
</div>"""

# ── Final HTML ────────────────────────────────────────────────────────────────
final_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Maharashtra Coastal Water Quality — Real-Data Dashboard</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.0/chart.umd.min.js"></script>
<style>
  :root {{
    --bg:#0b0f19; --panel:#121826; --card:#1a2233; --border:#232d42;
    --text:#eef2fb; --muted:#8892a6; --accent:#4c8dff; --accent2:#37c9a1;
    --danger:#ef5468; --warn:#f2a541; --radius:14px;
  }}
  * {{ margin:0; padding:0; box-sizing:border-box; }}
  body {{ font-family:'Segoe UI',system-ui,Arial,sans-serif; background:var(--bg); color:var(--text);
          height:100vh; display:flex; flex-direction:column; overflow:hidden; }}
  .header {{ background:linear-gradient(120deg,#101728,#182238); padding:14px 24px; display:flex;
             align-items:center; justify-content:space-between; border-bottom:1px solid var(--border); flex-shrink:0; }}
  .header-title {{ font-size:18px; font-weight:600; }}
  .header-sub {{ font-size:11px; color:var(--muted); margin-top:3px; }}
  .header-badges {{ display:flex; gap:8px; flex-wrap:wrap; }}
  .badge {{ padding:5px 12px; border-radius:999px; font-size:11px; font-weight:600; border:1px solid; }}
  .main {{ display:flex; flex:1; overflow:hidden; }}
  .map-container {{ flex:1; position:relative; }}
  .map-container iframe {{ width:100%; height:100%; border:none; }}
  .right-panel {{ width:440px; display:flex; flex-direction:column; background:var(--panel);
                  border-left:1px solid var(--border); flex-shrink:0; }}
  .tabs {{ display:flex; border-bottom:1px solid var(--border); flex-shrink:0; flex-wrap:wrap; }}
  .tab {{ flex:1; min-width:70px; padding:10px 6px; text-align:center; cursor:pointer; font-size:11px;
          font-weight:600; color:var(--muted); border-bottom:2px solid transparent; transition:all .15s; }}
  .tab.active {{ color:var(--accent); border-bottom-color:var(--accent); background:rgba(76,141,255,0.08); }}
  .tab:hover {{ color:var(--text); }}
  .tab-content {{ flex:1; overflow:hidden; display:none; flex-direction:column; }}
  .tab-content.active {{ display:flex; }}
  .scroll {{ flex:1; overflow-y:auto; padding:14px; }}
  .stat-card {{ background:var(--card); border:1px solid var(--border); border-radius:var(--radius);
                padding:14px; margin-bottom:12px; }}
  .stat-card-title {{ font-size:12px; font-weight:700; color:var(--accent); margin-bottom:10px;
                       text-transform:uppercase; letter-spacing:.03em; }}
  .location-row {{ display:flex; align-items:center; justify-content:space-between; padding:9px 0;
                    border-bottom:1px solid var(--border); }}
  .location-row:last-child {{ border:none; }}
  .loc-name {{ font-size:12px; font-weight:700; min-width:90px; }}
  .loc-bar-wrap {{ flex:1; margin:0 10px; }}
  .loc-bar {{ height:8px; border-radius:4px; background:var(--border); overflow:hidden; }}
  .loc-bar-fill {{ height:100%; border-radius:4px; }}
  .loc-grade {{ font-size:15px; font-weight:800; width:22px; text-align:center; }}
  .chat-messages {{ flex:1; overflow-y:auto; padding:14px; display:flex; flex-direction:column; gap:10px; }}
  .msg {{ max-width:92%; padding:10px 13px; border-radius:14px; font-size:12.5px; line-height:1.6; }}
  .msg.user {{ background:var(--accent); color:white; align-self:flex-end; border-radius:14px 14px 4px 14px; }}
  .msg.ai {{ background:var(--card); color:var(--text); align-self:flex-start; border:1px solid var(--border);
             border-radius:14px 14px 14px 4px; }}
  .msg.ai .ai-label {{ color:var(--accent2); font-weight:700; font-size:10px; margin-bottom:4px; }}
  .msg.loading {{ background:var(--card); color:var(--muted); font-style:italic; }}
  .chat-input-area {{ padding:12px; border-top:1px solid var(--border); flex-shrink:0; }}
  .chat-input-row {{ display:flex; gap:8px; margin-bottom:8px; }}
  .chat-input {{ flex:1; background:var(--card); border:1px solid var(--border); color:var(--text);
                 padding:10px 12px; border-radius:10px; font-size:12.5px; outline:none; resize:none; height:56px; }}
  .chat-input:focus {{ border-color:var(--accent); }}
  .send-btn {{ background:var(--accent); color:white; border:none; padding:0 18px; border-radius:10px;
               cursor:pointer; font-size:15px; }}
  .send-btn:hover {{ opacity:.88; }}
  .quick-btns {{ display:flex; flex-wrap:wrap; gap:5px; }}
  .quick-btn {{ background:var(--card); border:1px solid var(--border); color:var(--muted); padding:5px 9px;
                border-radius:8px; cursor:pointer; font-size:10px; }}
  .quick-btn:hover {{ color:var(--text); border-color:var(--accent); }}
  .data-table {{ width:100%; border-collapse:collapse; font-size:11px; }}
  .data-table th {{ text-align:left; color:var(--muted); padding:5px 6px; border-bottom:1px solid var(--border); }}
  .data-table td {{ padding:5px 6px; border-bottom:1px solid rgba(255,255,255,0.04); }}
  .data-table tr.best {{ background:rgba(55,201,161,0.1); }}
  .data-table.small th, .data-table.small td {{ padding:3px 4px; font-size:9.5px; text-align:center; }}
  .data-table.small td.hi {{ color:var(--accent2); }}
  .data-table.small td.lo {{ color:var(--warn); }}
  .data-table.small td.zero {{ color:#3a4257; }}
  .note {{ font-size:10.5px; color:var(--muted); margin-top:8px; line-height:1.5; }}
  .provenance {{ display:inline-block; font-size:8.5px; padding:2px 6px; border-radius:6px; margin-left:6px; }}
  .prov-measured {{ background:rgba(55,201,161,.15); color:var(--accent2); }}
  .prov-forecast {{ background:rgba(76,141,255,.15); color:var(--accent); }}
  .prov-ai {{ background:rgba(242,165,65,.15); color:var(--warn); }}
  .cause-row, .sol-card {{ margin:8px 0; }}
  .cause-label {{ display:flex; justify-content:space-between; font-size:11px; color:var(--muted); margin-bottom:3px; }}
  .cause-bar {{ height:6px; background:var(--border); border-radius:3px; overflow:hidden; }}
  .cause-fill {{ height:100%; border-radius:3px; background:var(--warn); }}
  .sol-card {{ padding:10px 12px; border-radius:0 10px 10px 0; background:rgba(76,141,255,.06); border-left:3px solid var(--accent); }}
  .sol-priority {{ font-size:9px; font-weight:700; color:var(--accent); }}
  .sol-title {{ font-size:12px; font-weight:700; margin:3px 0; }}
  .sol-desc {{ font-size:10.5px; color:var(--muted); line-height:1.4; }}
  .loc-select {{ display:flex; gap:6px; flex-wrap:wrap; padding:12px 14px 0; }}
  .loc-chip {{ padding:6px 12px; border-radius:999px; font-size:11px; font-weight:600; cursor:pointer;
               border:1px solid var(--border); color:var(--muted); }}
  .loc-chip.active {{ background:var(--accent); color:white; border-color:var(--accent); }}
  .refresh-btn {{ font-size:10px; color:var(--accent); cursor:pointer; text-decoration:underline; float:right; }}
  ::-webkit-scrollbar {{ width:5px; }}
  ::-webkit-scrollbar-thumb {{ background:var(--border); border-radius:3px; }}
</style>
</head>
<body>
<div class="header">
  <div>
    <div class="header-title">Maharashtra Coastal Water Quality — Real-Data Dashboard</div>
    <div class="header-sub">Sentinel-2 2021-2025 (real, verified) · RF+GB+DNN ensemble ({classifier_metrics[classifier_metrics.model=='Ensemble_MajorityVote'].accuracy.iloc[0]*100:.1f}% test accuracy) · Forecast 2026-2028 · Gemini AI</div>
  </div>
  <div class="header-badges">
    {''.join(f'<div class="badge" style="border-color:{grade_colors[overall[l]["grade"]]};color:{grade_colors[overall[l]["grade"]]}">{display_names[l]} · Grade {overall[l]["grade"]}</div>' for l in coords)}
  </div>
</div>
<div class="main">
  <div class="map-container">{map_html}</div>
  <div class="right-panel">
    <div class="tabs">
      <div class="tab active" onclick="switchTab('ai')">AI Chat</div>
      <div class="tab" onclick="switchTab('stats')">Timeline</div>
      <div class="tab" onclick="switchTab('causes')">Causes</div>
      <div class="tab" onclick="switchTab('solutions')">Solutions</div>
      <div class="tab" onclick="switchTab('method')">Methodology</div>
      <div class="tab" onclick="switchTab('quality')">Data QA</div>
    </div>

    <div class="tab-content active" id="tab-ai">
      <div class="chat-messages" id="chat-messages">
        <div class="msg ai"><div class="ai-label">Gemini Environmental Analyst</div>
        Hello! I have access to the real, verified Maharashtra coastal water quality data (2021-2025)
        and real 2026-2028 forecasts. Ask me anything — I'll answer from the actual computed numbers,
        including model limitations.</div>
      </div>
      <div class="chat-input-area">
        <div class="chat-input-row">
          <textarea class="chat-input" id="chat-input" placeholder="Ask about pollution, trends, forecasts, model accuracy..."
            onkeydown="if(event.key==='Enter'&&!event.shiftKey){{event.preventDefault();sendMessage()}}"></textarea>
          <button class="send-btn" onclick="sendMessage()">-></button>
        </div>
        <div class="quick-btns">
          <button class="quick-btn" onclick="ask('How accurate is the classifier really?')">Real accuracy?</button>
          <button class="quick-btn" onclick="ask('Why is Navi Mumbai so polluted?')">Navi Mumbai</button>
          <button class="quick-btn" onclick="ask('What will happen by 2028?')">2028 outlook</button>
          <button class="quick-btn" onclick="ask('Which forecasting model won for each location and why?')">Model selection</button>
          <button class="quick-btn" onclick="ask('What are the limitations of this analysis?')">Limitations</button>
        </div>
      </div>
    </div>

    <div class="tab-content" id="tab-stats">
      <div class="scroll">
        <div class="stat-card"><div class="stat-card-title">Overall risk (2021-2025 real average)</div>{location_rows_html}</div>
        <div class="stat-card">
          <div class="stat-card-title">Trend + forecast per location <span class="provenance prov-measured">measured</span><span class="provenance prov-forecast">forecast</span></div>
          <div class="loc-select" id="chart-loc-select" style="padding:0 0 10px 0">
            {make_loc_chips('showChart')}
          </div>
          <canvas id="timelineChart" height="180"></canvas>
        </div>
      </div>
    </div>

    <div class="tab-content" id="tab-causes">
      <div class="loc-select" id="causes-loc-select">
        {make_loc_chips('loadCauses')}
      </div>
      <div class="scroll" id="causes-content">
        <div class="note">Loading real-time analysis from Gemini, seeded with the real computed metrics for this location...</div>
      </div>
    </div>

    <div class="tab-content" id="tab-solutions">
      <div class="loc-select" id="solutions-loc-select">
        {make_loc_chips('loadSolutions')}
      </div>
      <div class="scroll" id="solutions-content">
        <div class="note">Loading real-time recommendations from Gemini...</div>
      </div>
    </div>

    <div class="tab-content" id="tab-method"><div class="scroll">{methodology_html}</div></div>
    <div class="tab-content" id="tab-quality"><div class="scroll">{data_quality_html}</div></div>
  </div>
</div>

<script>
const AI_CONTEXT = {ai_context_json};
const TIMELINE_DATA = {json.dumps(timeline_chart_data)};
const DISPLAY_NAMES = {json.dumps(display_names)};
let chatHistory = [];
let currentChart = null;
let csCache = {{}};

function switchTab(tab) {{
  const ids = ['ai','stats','causes','solutions','method','quality'];
  document.querySelectorAll('.tab').forEach((t,i) => t.classList.toggle('active', ids[i]===tab));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  document.getElementById('tab-'+tab).classList.add('active');
  if (tab==='stats' && !currentChart) showChart('{list(coords.keys())[0]}');
  if (tab==='causes' && !csCache['{list(coords.keys())[0]}_causes']) loadCauses('{list(coords.keys())[0]}');
  if (tab==='solutions' && !csCache['{list(coords.keys())[0]}_solutions']) loadSolutions('{list(coords.keys())[0]}');
}}

function showChart(loc) {{
  document.querySelectorAll('#chart-loc-select .loc-chip').forEach(c => c.classList.toggle('active', c.dataset.loc===loc));
  const d = TIMELINE_DATA[loc];
  if (currentChart) currentChart.destroy();
  const ctx = document.getElementById('timelineChart').getContext('2d');
  currentChart = new Chart(ctx, {{
    type: 'line',
    data: {{ labels: d.labels, datasets: [
      {{ label:'Polluted %', data:d.polluted, borderColor:'#EE5253', backgroundColor:'rgba(238,82,83,0.1)', tension:0.3, fill:true }},
      {{ label:'Clean %', data:d.clean, borderColor:'#2E86DE', backgroundColor:'rgba(46,134,222,0.08)', tension:0.3, fill:true }}
    ]}},
    options: {{ responsive:true, plugins:{{legend:{{labels:{{color:'#8892a6',font:{{size:10}}}}}}}},
      scales:{{ x:{{ticks:{{color:'#8892a6',font:{{size:9}}}}, grid:{{color:'#232d42'}}}},
                y:{{ticks:{{color:'#8892a6',font:{{size:9}}}}, grid:{{color:'#232d42'}}, min:0}} }} }}
  }});
}}

async function loadCauses(loc, refresh) {{
  document.querySelectorAll('#causes-loc-select .loc-chip').forEach(c => c.classList.toggle('active', c.dataset.loc===loc));
  const el = document.getElementById('causes-content');
  const cacheKey = loc+'_causes';
  if (!refresh && csCache[cacheKey]) {{ el.innerHTML = csCache[cacheKey]; return; }}
  el.innerHTML = '<div class="note">Generating real-time analysis for '+DISPLAY_NAMES[loc]+' from Gemini...</div>';
  try {{
    const r = await fetch('/causes_solutions/'+loc+(refresh?'?refresh=1':''));
    const data = await r.json();
    if (data.error && (!data.causes || !data.causes.length)) {{ el.innerHTML = '<div class="note">'+data.error+'</div>'; return; }}
    let html = '<div class="stat-card"><div class="stat-card-title">Pollution causes <span class="provenance prov-ai">gemini live</span><span class="refresh-btn" onclick="loadCauses(\\''+loc+'\\', true)">refresh</span></div>';
    (data.causes||[]).forEach(c => {{
      html += '<div class="cause-row"><div class="cause-label"><span>'+c.name+'</span><span>'+c.pct+'%</span></div>'+
              '<div class="cause-bar"><div class="cause-fill" style="width:'+c.pct+'%"></div></div>'+
              '<div class="note">'+c.desc+'</div></div>';
    }});
    html += '</div>';
    el.innerHTML = html; csCache[cacheKey] = html;
  }} catch(e) {{ el.innerHTML = '<div class="note">Error loading: '+e.message+'</div>'; }}
}}

async function loadSolutions(loc, refresh) {{
  document.querySelectorAll('#solutions-loc-select .loc-chip').forEach(c => c.classList.toggle('active', c.dataset.loc===loc));
  const el = document.getElementById('solutions-content');
  const cacheKey = loc+'_solutions';
  if (!refresh && csCache[cacheKey]) {{ el.innerHTML = csCache[cacheKey]; return; }}
  el.innerHTML = '<div class="note">Generating real-time recommendations for '+DISPLAY_NAMES[loc]+' from Gemini...</div>';
  try {{
    const r = await fetch('/causes_solutions/'+loc+(refresh?'?refresh=1':''));
    const data = await r.json();
    if (data.error && (!data.solutions || !data.solutions.length)) {{ el.innerHTML = '<div class="note">'+data.error+'</div>'; return; }}
    let html = '<div class="stat-card"><div class="stat-card-title">Recommended actions <span class="provenance prov-ai">gemini live</span><span class="refresh-btn" onclick="loadSolutions(\\''+loc+'\\', true)">refresh</span></div>';
    (data.solutions||[]).forEach(s => {{
      html += '<div class="sol-card"><div class="sol-priority">'+s.priority+'</div><div class="sol-title">'+s.title+'</div><div class="sol-desc">'+s.desc+'</div></div>';
    }});
    html += '</div>';
    el.innerHTML = html; csCache[cacheKey] = html;
  }} catch(e) {{ el.innerHTML = '<div class="note">Error loading: '+e.message+'</div>'; }}
}}

function ask(q) {{ document.getElementById('chat-input').value = q; sendMessage(); }}

function appendMsg(text, role) {{
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  if (role === 'ai') div.innerHTML = '<div class="ai-label">Gemini Environmental Analyst</div>' + text.replace(/\\n/g,'<br>');
  else div.textContent = text;
  container.appendChild(div); container.scrollTop = container.scrollHeight;
  return div;
}}

async function sendMessage() {{
  const input = document.getElementById('chat-input');
  const text = input.value.trim();
  if (!text) return;
  input.value = ''; appendMsg(text, 'user');
  chatHistory.push({{role:'user', parts:[{{text:text}}]}});
  const loadingDiv = appendMsg('Analysing real coastal data...', 'loading');
  try {{
    const contents = chatHistory.length === 1
      ? [{{role:'user', parts:[{{text: AI_CONTEXT + '\\n\\nUser question: ' + text}}]}}]
      : [{{role:'user', parts:[{{text: AI_CONTEXT}}]}},
         {{role:'model', parts:[{{text:'Understood. I have the real, verified water quality data and will answer honestly from it.'}}]}},
         ...chatHistory.slice(0,-1), {{role:'user', parts:[{{text:text}}]}}];
    const response = await fetch('/gemini', {{method:'POST', headers:{{'Content-Type':'application/json'}}, body:JSON.stringify({{contents}})}});
    const data = await response.json();
    if (data.error) throw new Error(data.error);
    const reply = data.candidates[0].content.parts[0].text;
    loadingDiv.remove(); appendMsg(reply, 'ai');
    chatHistory.push({{role:'model', parts:[{{text:reply}}]}});
  }} catch(err) {{ loadingDiv.remove(); appendMsg('Error: ' + err.message, 'ai'); console.error(err); }}
}}
</script>
</body>
</html>"""

out = os.path.join(HTML_PATH, "ai_water_quality_dashboard.html")
with open(out, 'w', encoding='utf-8') as f:
    f.write(final_html)

print("Dashboard HTML saved (real data, real-time Gemini causes/solutions).")
print("Starting Flask server...")
print("Open in Chrome: http://localhost:5000")
app.run(host='0.0.0.0', port=5000, debug=False)