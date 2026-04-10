"""
Export ESCO mapping to an interactive HTML viewer.
Usage: python testing/export_esco_mapping_viewer.py
Output: testing/esco_mapping_viewer.html  (open in browser)
"""
import json
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
MAPPING_FILE = BASE_DIR / "data" / "processed" / "jd_cv_pairs" / "skill_to_esco_mapping.json"
OUTPUT_HTML = Path(__file__).parent / "esco_mapping_viewer.html"


def main():
    print(f"Loading {MAPPING_FILE}...")
    with open(MAPPING_FILE, "r", encoding="utf-8") as f:
        mapping = json.load(f)

    # Sort by best similarity score descending
    rows = []
    for raw_skill, matches in mapping.items():
        if not matches:
            rows.append({"raw": raw_skill, "label": "—", "skill_id": "", "similarity": 0})
            continue
        best = matches[0]
        rows.append({
            "raw": raw_skill,
            "label": best.get("label", ""),
            "skill_id": best.get("skill_id", ""),
            "similarity": best.get("similarity", 0),
            "alt_matches": matches[1:],
        })

    rows.sort(key=lambda x: x["similarity"], reverse=True)

    # Serialize for JS
    rows_json = json.dumps(rows, ensure_ascii=False)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>ESCO Mapping Viewer</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f4f6f9; color: #333; }}
  header {{ background: #1e3a5f; color: white; padding: 16px 24px; display: flex; align-items: center; gap: 16px; flex-wrap: wrap; }}
  header h1 {{ font-size: 1.2rem; flex: 1 1 200px; }}
  .controls {{ display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }}
  .controls label {{ font-size: 0.85rem; color: #ccc; }}
  .controls input[type=range] {{ width: 160px; cursor: pointer; }}
  .controls input[type=text] {{ padding: 6px 10px; border-radius: 6px; border: none; font-size: 0.9rem; width: 220px; }}
  #threshold-display {{ font-size: 1.1rem; font-weight: bold; color: #ffd700; min-width: 40px; }}
  #stats {{ font-size: 0.85rem; color: #aac; margin-left: 8px; }}

  .table-wrap {{ overflow-x: auto; padding: 16px; }}
  table {{ width: 100%; border-collapse: collapse; background: white; border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px #0001; }}
  thead tr {{ background: #1e3a5f; color: white; }}
  th {{ padding: 10px 14px; text-align: left; font-size: 0.85rem; white-space: nowrap; }}
  td {{ padding: 9px 14px; font-size: 0.85rem; border-bottom: 1px solid #eee; vertical-align: top; }}
  tr:hover td {{ background: #f0f4ff; }}
  tr.hidden {{ display: none; }}

  .score-badge {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 0.82rem;
    color: white;
    min-width: 52px;
    text-align: center;
  }}
  .alt-matches {{ font-size: 0.78rem; color: #888; margin-top: 4px; }}
  .alt-matches span {{ display: inline-block; background: #eee; border-radius: 4px; padding: 1px 6px; margin: 2px 2px 0 0; }}

  .sort-btn {{ cursor: pointer; user-select: none; }}
  .sort-btn::after {{ content: " ⇅"; color: #aaa; font-size: 0.75rem; }}
  th.sorted-asc::after {{ content: " ↑"; color: #ffd700; }}
  th.sorted-desc::after {{ content: " ↓"; color: #ffd700; }}

  select {{ padding: 6px 10px; border-radius: 6px; border: none; font-size: 0.9rem; }}
</style>
</head>
<body>

<header>
  <h1>🔍 ESCO Skill Mapping Viewer</h1>
  <div class="controls">
    <label>Min score:
      <input type="range" id="threshold-slider" min="0.3" max="1.0" step="0.01" value="0.0">
    </label>
    <span id="threshold-display">0.00</span>
    <span id="stats"></span>
  </div>
  <div class="controls">
    <input type="text" id="search-box" placeholder="Search raw skill or ESCO label...">
    <select id="range-filter">
      <option value="">All ranges</option>
      <option value="0.9">≥ 0.90 (Very high)</option>
      <option value="0.8">≥ 0.80</option>
      <option value="0.7">≥ 0.70</option>
      <option value="0.65">≥ 0.65 (recommended)</option>
      <option value="0.6">≥ 0.60</option>
      <option value="0.5">≥ 0.50</option>
    </select>
  </div>
</header>

<div class="table-wrap">
  <table id="main-table">
    <thead>
      <tr>
        <th>#</th>
        <th class="sort-btn" data-col="raw">Raw Skill</th>
        <th class="sort-btn" data-col="label">ESCO Label</th>
        <th class="sort-btn" data-col="skill_id">ESCO ID</th>
        <th class="sort-btn sorted-desc" data-col="similarity">Score</th>
        <th>Alt Matches</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
</div>

<script>
const ALL_ROWS = {rows_json};

function scoreColor(s) {{
  if (s >= 0.8)  return '#2e7d32';
  if (s >= 0.7)  return '#558b2f';
  if (s >= 0.65) return '#f57f17';
  if (s >= 0.6)  return '#e65100';
  return '#b71c1c';
}}

let currentSort = {{ col: 'similarity', dir: -1 }};
let rows = [...ALL_ROWS];

function render(data) {{
  const tbody = document.getElementById('tbody');
  tbody.innerHTML = '';
  data.forEach((row, i) => {{
    const sim = row.similarity;
    const alts = (row.alt_matches || []).map(m =>
      `<span>${{m.label}} (${{m.similarity.toFixed(3)}})</span>`
    ).join('');
    const tr = document.createElement('tr');
    tr.dataset.sim = sim;
    tr.innerHTML = `
      <td style="color:#999">${{i+1}}</td>
      <td><strong>${{escHtml(row.raw)}}</strong></td>
      <td>${{escHtml(row.label)}}</td>
      <td style="font-size:0.75rem;color:#888">${{escHtml(row.skill_id || '')}}</td>
      <td><span class="score-badge" style="background:${{scoreColor(sim)}}">${{sim.toFixed(3)}}</span></td>
      <td><div class="alt-matches">${{alts}}</div></td>
    `;
    tbody.appendChild(tr);
  }});
  updateStats(data.length);
}}

function escHtml(s) {{
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}

function updateStats(shown) {{
  document.getElementById('stats').textContent = `${{shown}} / ${{ALL_ROWS.length}} skills`;
}}

function applyFilters() {{
  const threshold = parseFloat(document.getElementById('threshold-slider').value);
  const search = document.getElementById('search-box').value.toLowerCase();
  const rangeVal = document.getElementById('range-filter').value;
  const minScore = rangeVal ? parseFloat(rangeVal) : threshold;
  document.getElementById('threshold-display').textContent = minScore.toFixed(2);

  let filtered = rows.filter(r => {{
    if (r.similarity < minScore) return false;
    if (search && !r.raw.toLowerCase().includes(search) && !r.label.toLowerCase().includes(search)) return false;
    return true;
  }});
  render(filtered);
}}

// Slider
document.getElementById('threshold-slider').addEventListener('input', function() {{
  document.getElementById('range-filter').value = '';
  applyFilters();
}});

// Search
document.getElementById('search-box').addEventListener('input', applyFilters);

// Range filter
document.getElementById('range-filter').addEventListener('change', function() {{
  if (this.value) {{
    document.getElementById('threshold-slider').value = this.value;
    document.getElementById('threshold-display').textContent = parseFloat(this.value).toFixed(2);
  }}
  applyFilters();
}});

// Sort
document.querySelectorAll('.sort-btn').forEach(th => {{
  th.addEventListener('click', function() {{
    const col = this.dataset.col;
    if (currentSort.col === col) {{
      currentSort.dir *= -1;
    }} else {{
      currentSort.col = col;
      currentSort.dir = col === 'similarity' ? -1 : 1;
    }}
    document.querySelectorAll('th').forEach(t => t.classList.remove('sorted-asc','sorted-desc'));
    this.classList.add(currentSort.dir === 1 ? 'sorted-asc' : 'sorted-desc');
    rows.sort((a, b) => {{
      const av = a[col] || '';
      const bv = b[col] || '';
      if (typeof av === 'number') return currentSort.dir * (av - bv);
      return currentSort.dir * av.localeCompare(bv, undefined, {{sensitivity: 'base'}});
    }});
    applyFilters();
  }});
}});

// Init
render(rows);
updateStats(rows.length);
</script>
</body>
</html>
"""

    OUTPUT_HTML.write_text(html, encoding="utf-8")
    print(f"✓ Exported {len(rows)} mappings → {OUTPUT_HTML}")
    print(f"  Open in browser: file://{OUTPUT_HTML.resolve()}")


if __name__ == "__main__":
    main()
