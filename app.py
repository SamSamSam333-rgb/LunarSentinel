from flask import Flask, render_template_string
import pickle

app = Flask(__name__)

with open("ice_clusters.pkl", "rb") as f:
    clusters = pickle.load(f)
with open("prob_map_shape.pkl", "rb") as f:
    img_h, img_w = pickle.load(f)
with open("feature_importance.pkl", "rb") as f:
    feature_importance = pickle.load(f)
with open("scatter_sample.pkl", "rb") as f:
    scatter_sample = pickle.load(f)

@app.route("/")
def dashboard():
    sorted_clusters = sorted(clusters, key=lambda c: -c['pixels'])
    markers_js = ",\n".join([
        f"""{{lat: {img_h - c['row']}, lng: {c['col']}, id: {c['cluster_id']}, px: {c['pixels']}, real_lat: {c['lat']:.4f}, real_lon: {c['lon']:.4f}}}"""
        for c in clusters
    ])

    table_rows = "\n".join([
        f"""<tr><td>{c['cluster_id']}</td><td>{c['pixels']}</td>
        <td>{c['lat']:.4f}&deg;</td><td>{c['lon']:.4f}&deg;</td>
        <td><span class="conf-tag">{'High' if c['pixels']>=8 else 'Moderate'}</span></td></tr>"""
        for c in sorted_clusters
    ])

    max_importance = max(v for _, v in feature_importance)
    importance_html = "\n".join([
        f"""<div class="bar-row">
            <div class="bar-label">{name}</div>
            <div class="bar-track"><div class="bar-fill" data-w="{(val/max_importance*100):.0f}"></div></div>
            <div class="bar-value">{val*100:.1f}%</div>
        </div>"""
        for name, val in feature_importance
    ])

    scatter_js = ",".join([
        f"{{cpr:{p['cpr']:.4f},srd:{p['srd']:.4f},p:{p['proba']:.4f}}}" for p in scatter_sample
    ])

    return render_template_string("""
    <html>
    <head>
        <title>Lunar Ice Sentinel</title>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.css"/>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/leaflet/1.9.4/leaflet.min.js"></script>
        <style>
            * { box-sizing: border-box; margin:0; padding:0; }
            body { font-family:'Segoe UI',sans-serif; background:#03040a; color:#e0f7fa; overflow-x:hidden; position:relative; }
            #stars { position:fixed; inset:0; z-index:0; }
            .content { position:relative; z-index:1; max-width:1400px; margin:0 auto; }

            .header { padding:24px 34px; background:linear-gradient(90deg,#03040a00,#0a1420cc 40%,#03040a00);
                backdrop-filter:blur(6px); border-bottom:1px solid #00e5ff33;
                display:flex; justify-content:space-between; align-items:center; }
            .header-left { display:flex; align-items:center; gap:16px; }
            .radar { width:36px; height:36px; border-radius:50%; border:2px solid #00e5ff; position:relative; flex-shrink:0; }
            .radar::before { content:''; position:absolute; inset:0; margin:auto; width:2px; height:50%; background:#00e5ff;
                top:0; left:50%; transform-origin:bottom center; animation:sweep 2.4s linear infinite; box-shadow:0 0 8px #00e5ff; }
            @keyframes sweep { 100% { transform: rotate(360deg); } }
            .header h1 { font-size:28px; letter-spacing:3px; background:linear-gradient(90deg,#e0f7fa,#00e5ff);
                -webkit-background-clip:text; background-clip:text; color:transparent; }
            .header p { margin-top:4px; color:#7d8b93; font-size:13px; letter-spacing:0.5px; }
            .badge { background:#0d2b30cc; border:1px solid #00e5ff66; color:#00e5ff; padding:7px 16px;
                border-radius:20px; font-size:12px; letter-spacing:1px; box-shadow:0 0 12px #00e5ff22; }

            .stats { display:flex; gap:16px; padding:20px 34px; flex-wrap:wrap; }
            .stat-card { background:#0d1620aa; backdrop-filter:blur(8px); border-radius:12px; padding:16px 24px;
                border:1px solid #1a2530; min-width:160px; transition:transform .2s, border-color .2s; }
            .stat-card:hover { transform:translateY(-3px); border-color:#00e5ff77; }
            .stat-card .label { font-size:11px; color:#7d8b93; text-transform:uppercase; letter-spacing:1px; }
            .stat-card .value { font-size:26px; font-weight:700; color:#00e5ff; margin-top:4px; text-shadow:0 0 12px #00e5ff55; }

            .hero { position:relative; margin:24px 34px; border-radius:16px; overflow:hidden;
                border:1px solid #1a2530; box-shadow:0 0 60px #00e5ff1a; }
            .hero img { width:100%; display:block; }
            .hero::after { content:''; position:absolute; left:0; right:0; height:3px; background:#00e5ffcc;
                box-shadow:0 0 16px 4px #00e5ff; animation:scanline 3.2s linear infinite; }
            @keyframes scanline { 0% { top:0; } 100% { top:100%; } }
            .hero-tag { position:absolute; top:18px; left:18px; background:#000b; padding:9px 18px;
                border-radius:8px; border:1px solid #00e5ff55; font-size:13px; letter-spacing:1px; backdrop-filter:blur(4px); }
            .hero-tag b { color:#00e5ff; }

            .section-title { padding:0 34px; margin:34px 0 14px; font-size:13px; letter-spacing:2px;
                color:#7d8b93; text-transform:uppercase; display:flex; align-items:center; gap:12px; }
            .section-title::after { content:''; flex:1; height:1px; background:linear-gradient(90deg,#1a2530,transparent); }

            .layout { display:flex; gap:16px; padding:0 34px 20px; }
            .map-panel { flex:2; position:relative; border-radius:14px; overflow:hidden; border:1px solid #1a2530; }
            #map { height:58vh; width:100%; background:#03040a; }
            .leaflet-container { background:#03040a !important; }
            .leaflet-popup-content-wrapper { background:#0d1620; color:#e0f7fa; border:1px solid #00e5ff44; border-radius:8px; }
            .leaflet-popup-tip { background:#0d1620; }
            .legend { position:absolute; bottom:14px; left:14px; z-index:500; background:#03040aee;
                border:1px solid #1a2530; border-radius:8px; padding:9px 14px; font-size:12px; color:#b8c4cc; }
            .legend .dot { display:inline-block; width:10px; height:10px; border-radius:50%; background:#00e5ff;
                margin-right:6px; box-shadow:0 0 6px #00e5ff; }

            .gallery { flex:1; display:flex; flex-direction:column; gap:14px; }
            .gallery-card { background:#0d1620aa; backdrop-filter:blur(6px); border:1px solid #1a2530; border-radius:12px;
                overflow:hidden; cursor:pointer; transition:border-color .15s, transform .15s; }
            .gallery-card:hover { border-color:#00e5ff88; transform:translateY(-2px); }
            .gallery-card img { width:100%; height:130px; object-fit:cover; display:block; }
            .gallery-card .cap { padding:10px 12px; }
            .gallery-card .cap-title { font-size:13px; font-weight:600; }
            .gallery-card .cap-sub { font-size:11px; color:#7d8b93; margin-top:2px; }

            .method-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:16px; padding:0 34px; }
            .method-card { background:#0d1620aa; backdrop-filter:blur(6px); border:1px solid #1a2530; border-radius:12px; padding:18px 20px; }
            .method-card .param-name { font-size:16px; font-weight:700; color:#00e5ff; }
            .method-card .param-full { font-size:11px; color:#7d8b93; margin-bottom:8px; }
            .method-card .param-desc { font-size:12.5px; color:#b8c4cc; line-height:1.5; }
            .method-card .param-rule { margin-top:10px; padding:8px 10px; background:#03040a; border-radius:6px;
                font-size:11.5px; color:#00e5ff; font-family:'Consolas',monospace; border:1px solid #1a2530; }

            .table-card { margin:0 34px 20px; background:#0d1620aa; backdrop-filter:blur(6px); border:1px solid #1a2530;
                border-radius:12px; padding:20px; overflow-x:auto; }
            table { width:100%; border-collapse:collapse; font-size:13px; }
            th { text-align:left; padding:8px 12px; color:#7d8b93; text-transform:uppercase; font-size:11px;
                letter-spacing:0.5px; border-bottom:1px solid #1a2530; }
            td { padding:9px 12px; border-bottom:1px solid #12181f; color:#e0f7fa; }
            tr:hover td { background:#00e5ff0a; }
            .conf-tag { background:#00e5ff22; color:#00e5ff; padding:3px 10px; border-radius:12px; font-size:11px; }

            .analysis-layout { display:flex; gap:16px; padding:0 34px 20px; }
            .chart-card { flex:1; background:#0d1620aa; backdrop-filter:blur(6px); border:1px solid #1a2530; border-radius:12px; padding:20px; }
            .chart-card h3 { font-size:13px; letter-spacing:1px; color:#7d8b93; text-transform:uppercase; margin-bottom:6px; }
            .chart-card .chart-sub { font-size:11.5px; color:#5c6b73; margin-bottom:14px; }
            .bar-row { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
            .bar-label { width:70px; font-size:12px; color:#b8c4cc; }
            .bar-track { flex:1; background:#0a0e14; border-radius:4px; height:16px; overflow:hidden; }
            .bar-fill { height:100%; background:linear-gradient(90deg,#00e5ff33,#00e5ff); border-radius:4px;
                box-shadow:0 0 8px #00e5ff88; width:0%; transition:width 1.2s ease-out; }
            .bar-value { width:44px; font-size:12px; color:#00e5ff; text-align:right; }

            #scatterSvg { width:100%; height:auto; }
            .axis-label { fill:#7d8b93; font-size:10px; }
            .grid-line { stroke:#1a2530; stroke-width:1; }

            .refs-card { margin:0 34px 40px; background:#0d1620aa; backdrop-filter:blur(6px); border:1px solid #1a2530;
                border-radius:12px; padding:20px; font-size:12px; color:#7d8b93; line-height:1.8; }
            .refs-card h3 { font-size:13px; letter-spacing:1px; color:#7d8b93; text-transform:uppercase; margin-bottom:10px; }
            .refs-card b { color:#b8c4cc; }

            .pulse-marker { border-radius:50%; background:#00e5ff; box-shadow:0 0 0 0 rgba(0,229,255,0.7); animation:pulse 2s infinite; }
            @keyframes pulse { 0%{box-shadow:0 0 0 0 rgba(0,229,255,.6);} 70%{box-shadow:0 0 0 14px rgba(0,229,255,0);} 100%{box-shadow:0 0 0 0 rgba(0,229,255,0);} }

            .lightbox { display:none; position:fixed; inset:0; background:rgba(0,0,0,0.92); z-index:2000;
                align-items:center; justify-content:center; }
            .lightbox.active { display:flex; }
            .lightbox img { max-width:92%; max-height:88%; border-radius:8px; border:1px solid #00e5ff55; }
            .lightbox .close-btn { position:absolute; top:20px; right:34px; font-size:32px; cursor:pointer; color:#e0f7fa; }
        </style>
    </head>
    <body>
        <canvas id="stars"></canvas>
        <div class="content">
            <div class="header">
                <div class="header-left">
                    <div class="radar"></div>
                    <div>
                        <h1>LUNAR ICE SENTINEL</h1>
                        <p>Shackleton Crater &middot; South Pole PSR Ice Detection via Full-Polarimetric DFSAR</p>
                    </div>
                </div>
                <div class="badge">Chandrayaan-2 &middot; DFSAR + OHRC</div>
            </div>

            <div class="stats">
                <div class="stat-card"><div class="label">Clusters detected</div><div class="value" id="statClusters">0</div></div>
                <div class="stat-card"><div class="label">Ice-candidate pixels</div><div class="value" id="statPixels">0</div></div>
                <div class="stat-card"><div class="label">Detection model</div><div class="value" style="font-size:18px;">RF ensemble</div></div>
                <div class="stat-card"><div class="label">Analysis area</div><div class="value" style="font-size:18px;">~80&times;80 km</div></div>
                <div class="stat-card"><div class="label">Spatial resolution</div><div class="value" style="font-size:18px;">~24 m/px</div></div>
            </div>

            <div class="hero" onclick="openLightbox('/static/rover_path.png')">
                <div class="hero-tag">Autonomous path &middot; <b>A* rover route</b> over OHRC-derived terrain cost map (0.25 m/px)</div>
                <img src="/static/rover_path.png">
            </div>

            <div class="section-title">Detection methodology</div>
            <div class="method-grid">
                <div class="method-card">
                    <div class="param-name">CPR</div>
                    <div class="param-full">Circular Polarization Ratio</div>
                    <div class="param-desc">Ratio of same-sense to opposite-sense circularly polarized radar return. Elevated values indicate multiple-bounce scattering consistent with volumetric ice or rough ejecta.</div>
                    <div class="param-rule">Threshold: CPR &gt; 1.0</div>
                </div>
                <div class="method-card">
                    <div class="param-name">SRD</div>
                    <div class="param-full">Single-bounce Eigenvalue Relative Difference</div>
                    <div class="param-desc">Eigenvalue-decomposition metric quantifying dominance of single-bounce scattering. Low values indicate departure from simple surface reflection.</div>
                    <div class="param-rule">Threshold: SRD &lt; 0.3</div>
                </div>
                <div class="method-card">
                    <div class="param-name">T-Ratio</div>
                    <div class="param-full">Polarimetric T-Ratio</div>
                    <div class="param-desc">Full-polarimetric decomposition parameter sensitive to subsurface dielectric contrast; used as a corroborating indicator alongside CPR/SRD.</div>
                    <div class="param-rule">Cross-validation parameter</div>
                </div>
            </div>

            <div class="section-title">Radar-derived ice probability map</div>
            <div class="layout">
                <div class="map-panel">
                    <div id="map"></div>
                    <div class="legend"><span class="dot"></span> Ice-candidate cluster &middot; marker size = pixel count</div>
                </div>
                <div class="gallery">
                    <div class="gallery-card" onclick="openLightbox('/static/ohrc_terrain.png')">
                        <img src="/static/ohrc_terrain.png">
                        <div class="cap"><div class="cap-title">OHRC illuminated terrain</div>
                        <div class="cap-sub">0.25 m/px real crater surface near the rim</div></div>
                    </div>
                    <div class="gallery-card" onclick="openLightbox('/static/ohrc_psr_floor.png')">
                        <img src="/static/ohrc_psr_floor.png">
                        <div class="cap"><div class="cap-title">OHRC PSR floor</div>
                        <div class="cap-sub">Zero-signal noise confirms permanent shadow</div></div>
                    </div>
                </div>
            </div>

            <div class="section-title">Detected cluster inventory</div>
            <div class="table-card">
                <table>
                    <tr><th>Cluster ID</th><th>Pixel count</th><th>Latitude</th><th>Longitude</th><th>Confidence</th></tr>
                    {{ table_rows|safe }}
                </table>
            </div>

            <div class="section-title">Model diagnostics &amp; feature-space analysis</div>
            <div class="analysis-layout">
                <div class="chart-card">
                    <h3>Feature importance</h3>
                    <div class="chart-sub">Relative contribution of each radar parameter to the Random Forest's ice classification</div>
                    {{ importance_html|safe }}
                </div>
                <div class="chart-card" style="flex:1.4;">
                    <h3>CPR vs SRD feature space</h3>
                    <div class="chart-sub">2,500 sampled pixels &middot; color = model-predicted ice probability &middot; this separation is the actual basis for detection</div>
                    <svg id="scatterSvg" viewBox="0 0 480 320"></svg>
                </div>
            </div>

            <div class="refs-card">
                <h3>Methodology reference</h3>
                <p><b>Putrevu, D. et al. (2023)</b> &mdash; "Lunar Impact Craters: New Perspectives From Full-Polarimetric Analysis of Chandrayaan-2 Dual-Frequency SAR Data," <i>Journal of Geophysical Research: Planets</i>, 128, e2023JE007745.</p>
                <p style="margin-top:8px;">Ice-candidate classification derives from a Random Forest ensemble trained on physics-based weak labels (CPR &gt; 1.0 &amp; SRD &lt; 0.3), following the diagnostic criteria established for full-polarimetric DFSAR analysis. Terrain cost mapping and rover path planning are derived from Chandrayaan-2 OHRC calibrated imagery (0.25 m/pixel), used as a roughness/illumination proxy for traversability &mdash; a simplification stated here as a limitation pending power-budget-weighted refinement.</p>
            </div>
        </div>

        <div class="lightbox" id="lightbox" onclick="closeLightbox()">
            <span class="close-btn">&times;</span>
            <img id="lightbox-img" src="">
        </div>

        <script>
            var canvas = document.getElementById('stars');
            var ctx = canvas.getContext('2d');
            function resize() { canvas.width = window.innerWidth; canvas.height = document.body.scrollHeight; }
            resize(); window.addEventListener('resize', resize);
            var stars = Array.from({length:180}, () => ({
                x: Math.random()*canvas.width, y: Math.random()*canvas.height,
                r: Math.random()*1.4+0.3, a: Math.random(), speed: Math.random()*0.3+0.05
            }));
            function drawStars() {
                ctx.clearRect(0,0,canvas.width,canvas.height);
                ctx.fillStyle = '#03040a'; ctx.fillRect(0,0,canvas.width,canvas.height);
                stars.forEach(s => {
                    s.a += s.speed*0.02;
                    var op = 0.3 + Math.abs(Math.sin(s.a))*0.7;
                    ctx.beginPath(); ctx.arc(s.x, s.y, s.r, 0, Math.PI*2);
                    ctx.fillStyle = 'rgba(224,247,250,'+op+')'; ctx.fill();
                });
                requestAnimationFrame(drawStars);
            }
            drawStars();

            function countUp(id, target) {
                var el = document.getElementById(id); var cur = 0;
                var step = Math.max(1, target/40);
                var iv = setInterval(() => {
                    cur += step;
                    if (cur >= target) { cur = target; clearInterval(iv); }
                    el.textContent = Math.floor(cur);
                }, 25);
            }
            countUp('statClusters', {{ n_clusters }});
            countUp('statPixels', {{ total_px }});

            var imgW = {{ img_w }}, imgH = {{ img_h }};
            var map = L.map('map', { crs: L.CRS.Simple, minZoom:-3, maxZoom:5, zoomSnap:0.25 });
            var bounds = [[0,0],[imgH, imgW]];
            L.imageOverlay('/static/ice_map_clean.png', bounds).addTo(map);
            map.fitBounds(bounds);
            map.setZoom(map.getZoom() + 1.2);

            var markers = [{{ markers_js|safe }}];
            markers.forEach(function(m) {
                var size = 12 + m.px * 0.7;
                var icon = L.divIcon({ className:'', html:'<div class="pulse-marker" style="width:'+size+'px;height:'+size+'px;"></div>', iconSize:[size,size] });
                var marker = L.marker([m.lat, m.lng], { icon: icon }).addTo(map);
                marker.bindPopup('<b>Cluster '+m.id+'</b><br>'+m.px+' pixels<br>Lat: '+m.real_lat+'<br>Lon: '+m.real_lon);
            });

            setTimeout(() => {
                document.querySelectorAll('.bar-fill').forEach(el => el.style.width = el.dataset.w + '%');
            }, 200);

            // CPR vs SRD scatter plot, rendered as SVG
            var scatterData = [{{ scatter_js|safe }}];
            var svg = document.getElementById('scatterSvg');
            var W = 480, H = 320, pad = 40;

            var cprVals = scatterData.map(d => d.cpr);
            var srdVals = scatterData.map(d => d.srd);
            var cprMax = Math.min(3, Math.max(...cprVals));
            var srdMax = Math.max(...srdVals);

            function xScale(v) { return pad + (v / cprMax) * (W - pad*1.5); }
            function yScale(v) { return H - pad - (v / srdMax) * (H - pad*1.5); }

            var svgParts = [];
            // grid lines
            for (var i=0; i<=4; i++) {
                var gx = pad + i*(W-pad*1.5)/4;
                svgParts.push(`<line class="grid-line" x1="${gx}" y1="${pad}" x2="${gx}" y2="${H-pad}"/>`);
            }
            for (var i=0; i<=4; i++) {
                var gy = pad + i*(H-pad*1.5)/4;
                svgParts.push(`<line class="grid-line" x1="${pad}" y1="${gy}" x2="${W-pad*0.5}" y2="${gy}"/>`);
            }
            // axes
            svgParts.push(`<line x1="${pad}" y1="${H-pad}" x2="${W-pad*0.5}" y2="${H-pad}" stroke="#3a4750" stroke-width="1.2"/>`);
            svgParts.push(`<line x1="${pad}" y1="${pad}" x2="${pad}" y2="${H-pad}" stroke="#3a4750" stroke-width="1.2"/>`);
            svgParts.push(`<text class="axis-label" x="${W/2}" y="${H-8}" text-anchor="middle">CPR</text>`);
            svgParts.push(`<text class="axis-label" x="12" y="${H/2}" text-anchor="middle" transform="rotate(-90 12 ${H/2})">SRD</text>`);

            // threshold reference lines (CPR=1, SRD=0.3)
            var thrX = xScale(1.0), thrY = yScale(0.3);
            svgParts.push(`<line x1="${thrX}" y1="${pad}" x2="${thrX}" y2="${H-pad}" stroke="#ff6b6b" stroke-width="1" stroke-dasharray="4 3" opacity="0.6"/>`);
            svgParts.push(`<line x1="${pad}" y1="${thrY}" x2="${W-pad*0.5}" y2="${thrY}" stroke="#ff6b6b" stroke-width="1" stroke-dasharray="4 3" opacity="0.6"/>`);

            // points
            scatterData.forEach(d => {
                var x = xScale(Math.min(d.cpr, cprMax));
                var y = yScale(d.srd);
                var opacity = 0.25 + d.p*0.7;
                var color = d.p > 0.5 ? '#00e5ff' : '#5c6b73';
                var r = d.p > 0.5 ? 3 : 1.6;
                svgParts.push(`<circle cx="${x}" cy="${y}" r="${r}" fill="${color}" opacity="${opacity}"/>`);
            });

            svg.innerHTML = svgParts.join('');

            function openLightbox(src) { document.getElementById('lightbox-img').src = src; document.getElementById('lightbox').classList.add('active'); }
            function closeLightbox() { document.getElementById('lightbox').classList.remove('active'); }
        </script>
    </body>
    </html>
    """, markers_js=markers_js, img_w=img_w, img_h=img_h,
         n_clusters=len(clusters), total_px=sum(c['pixels'] for c in clusters),
         table_rows=table_rows, importance_html=importance_html, scatter_js=scatter_js)

if __name__ == "__main__":
    app.run(debug=True, port=5000)