"""Interactive Web Cockpit for Smart Meter Anomaly Inspection and Diagnosis.

Runs an integrated, zero-dependency HTTP server inspired by VETOR (SanPlat AMI System).
Serves:
1. REST APIs:
   - `/api/v1/summary`: Dataset statistics, category counts, model provenance.
   - `/api/v1/anomalies`: Paged, searchable, and filtered table of classified meters.
   - `/api/v1/meter/<uc>`: Detailed multi-feature diagnostics, time-series curves, and root-cause breakdown for a single UC.
   - `/api/v1/categories`: List of identified diagnostic categories and counts.
   - `/api/v1/feeders`: Anomaly rates by electrical feeder.
2. Web UI: Modern React-style SPA with live search, category pills, feeder filter, detailed drawer, and export.
"""

from __future__ import annotations

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import mimetypes
from pathlib import Path
import sys
import urllib.parse
import webbrowser

import polars as pl

PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASE_DIR = PROJECT_ROOT / "output"
REPORTS_DIR = BASE_DIR / "reports"
CLASSIFIED_PARQUET = REPORTS_DIR / "classified_ucs_latest.parquet"
CLASSIFIED_CSV = REPORTS_DIR / "classified_ucs_latest.csv"
SUMMARY_JSON = REPORTS_DIR / "model_training_summary_latest.json"


def _load_data() -> pl.DataFrame:
    """Loads latest classified UCs dataset transparently."""
    if CLASSIFIED_PARQUET.exists() and CLASSIFIED_PARQUET.stat().st_size > 0:
        return pl.read_parquet(CLASSIFIED_PARQUET)
    if CLASSIFIED_CSV.exists() and CLASSIFIED_CSV.stat().st_size > 0:
        return pl.read_csv(CLASSIFIED_CSV, separator=";", infer_schema_length=5000, ignore_errors=True)
    return pl.DataFrame()


def _load_summary() -> dict:
    if SUMMARY_JSON.exists():
        try:
            with open(SUMMARY_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Cockpit Analítico AMI — Detecção de Anomalias Copel Araucária</title>
  <style>
    :root {
      --bg: #0b0f17;
      --card-bg: #161d26;
      --card-border: #232d3b;
      --text: #f1f5f9;
      --text-dim: #94a3b8;
      --primary: #3b82f6;
      --primary-hover: #2563eb;
      --danger: #ef4444;
      --warning: #f59e0b;
      --success: #10b981;
      --accent: #8b5cf6;
      --solar: #eab308;
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      background-color: var(--bg);
      color: var(--text);
      line-height: 1.5;
      padding-bottom: 40px;
    }
    header {
      background: linear-gradient(180deg, #1e2631 0%, #161d26 100%);
      border-bottom: 1px solid var(--card-border);
      padding: 16px 28px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      position: sticky;
      top: 0;
      z-index: 100;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
    }
    .brand-icon {
      width: 36px;
      height: 36px;
      border-radius: 8px;
      background: linear-gradient(135deg, var(--primary), #1d4ed8);
      display: flex;
      align-items: center;
      justify-content: center;
      font-size: 18px;
      font-weight: bold;
    }
    .brand-title { font-size: 17px; font-weight: 700; }
    .brand-subtitle { font-size: 11px; color: var(--text-dim); }
    .header-actions { display: flex; gap: 10px; align-items: center; }
    .btn {
      background-color: var(--primary);
      color: white;
      border: none;
      padding: 8px 16px;
      border-radius: 6px;
      font-size: 13px;
      font-weight: 600;
      cursor: pointer;
      text-decoration: none;
      display: inline-flex;
      align-items: center;
      gap: 6px;
    }
    .btn:hover { background-color: var(--primary-hover); }
    .btn-secondary { background-color: #242f3d; color: var(--text); }
    .btn-secondary:hover { background-color: #334155; }
    
    .container { max-width: 1440px; margin: 0 auto; padding: 24px; }
    
    /* Metrics Row */
    .metrics-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 16px;
      margin-bottom: 24px;
    }
    .metric-card {
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 6px;
    }
    .metric-label { font-size: 12px; color: var(--text-dim); font-weight: 600; text-transform: uppercase; }
    .metric-value { font-size: 26px; font-weight: 800; }
    .metric-sub { font-size: 11px; color: var(--text-dim); }
    
    /* Filters Bar */
    .filter-panel {
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      padding: 16px;
      margin-bottom: 20px;
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      align-items: center;
      justify-content: space-between;
    }
    .search-box {
      flex: 1;
      min-width: 260px;
      position: relative;
    }
    .search-input {
      width: 100%;
      background-color: #0b0f17;
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 9px 12px 9px 34px;
      border-radius: 6px;
      font-size: 13px;
    }
    .search-input:focus { outline: none; border-color: var(--primary); }
    .search-icon {
      position: absolute;
      left: 10px;
      top: 50%;
      transform: translateY(-50%);
      color: var(--text-dim);
      font-size: 14px;
    }
    .select-filter {
      background-color: #0b0f17;
      border: 1px solid var(--card-border);
      color: var(--text);
      padding: 9px 12px;
      border-radius: 6px;
      font-size: 13px;
    }
    
    /* Table */
    .table-card {
      background-color: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 8px;
      overflow: hidden;
    }
    .table-responsive { width: 100%; overflow-x: auto; }
    table { width: 100%; border-collapse: collapse; text-align: left; font-size: 13px; }
    th {
      background-color: #121820;
      color: var(--text-dim);
      font-weight: 600;
      padding: 12px 14px;
      border-bottom: 1px solid var(--card-border);
      white-space: nowrap;
    }
    td {
      padding: 12px 14px;
      border-bottom: 1px solid #1a222d;
      white-space: nowrap;
    }
    tr:hover td { background-color: rgba(59, 130, 246, 0.05); }
    
    .badge {
      display: inline-block;
      padding: 3px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.3px;
    }
    .badge-fraud { background-color: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-solar { background-color: rgba(234, 179, 8, 0.15); color: #facc15; border: 1px solid rgba(234, 179, 8, 0.3); }
    .badge-tamper { background-color: rgba(139, 92, 246, 0.15); color: #c084fc; border: 1px solid rgba(139, 92, 246, 0.3); }
    .badge-telecom { background-color: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid rgba(148, 163, 184, 0.3); }
    .badge-meter { background-color: rgba(249, 115, 22, 0.15); color: #fb923c; border: 1px solid rgba(249, 115, 22, 0.3); }
    .badge-regular { background-color: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.3); }

    .score-bar-bg { width: 60px; height: 6px; background-color: #222d3d; border-radius: 3px; display: inline-block; margin-right: 8px; vertical-align: middle; overflow: hidden; }
    .score-bar-fill { height: 100%; border-radius: 3px; }
    
    .pagination {
      padding: 12px 16px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      background-color: #121820;
      border-top: 1px solid var(--card-border);
      font-size: 12px;
      color: var(--text-dim);
    }
    
    /* Drawer */
    .drawer {
      position: fixed;
      right: -520px;
      top: 0;
      width: 500px;
      height: 100vh;
      background-color: var(--card-bg);
      border-left: 1px solid var(--card-border);
      box-shadow: -8px 0 24px rgba(0,0,0,0.5);
      transition: right 0.3s ease;
      z-index: 200;
      overflow-y: auto;
      padding: 24px;
    }
    .drawer.open { right: 0; }
    .drawer-overlay {
      position: fixed;
      top: 0; left: 0; width: 100vw; height: 100vh;
      background: rgba(0,0,0,0.6);
      z-index: 199;
      display: none;
    }
    .drawer-overlay.open { display: block; }
    .drawer-header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 12px;
    }
    .drawer-close { background: none; border: none; color: var(--text-dim); font-size: 22px; cursor: pointer; }
    .detail-section { margin-bottom: 20px; }
    .detail-title { font-size: 12px; color: var(--text-dim); font-weight: 700; text-transform: uppercase; margin-bottom: 8px; }
    .kv-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; font-size: 13px; }
    .kv-box { background: #0b0f17; padding: 10px; border-radius: 6px; border: 1px solid var(--card-border); }
    .kv-k { font-size: 11px; color: var(--text-dim); }
    .kv-v { font-weight: 700; margin-top: 2px; }
  </style>
</head>
<body>

  <header>
    <div class="brand">
      <div class="brand-icon">⚡</div>
      <div>
        <div class="brand-title">COCKPIT ANALÍTICO AMI — COPEL ARAUCÁRIA</div>
        <div class="brand-subtitle">Monitoramento Inteligente de Perdas Não-Técnicas, GD e Qualidade de Energia</div>
      </div>
    </div>
    <div class="header-actions">
      <button class="btn btn-secondary" onclick="exportData('csv')">📥 Exportar CSV</button>
      <button class="btn" onclick="fetchData()">🔄 Atualizar Dados</button>
    </div>
  </header>

  <div class="container">
    <!-- Top KPI Cards -->
    <div class="metrics-grid" id="kpiGrid">
      <div class="metric-card">
        <span class="metric-label">Total UCs Analisadas</span>
        <span class="metric-value" id="kpiTotal">-</span>
        <span class="metric-sub">Araucária Completa</span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Anomalias Confirmadas</span>
        <span class="metric-value" style="color: var(--danger)" id="kpiAnom">-</span>
        <span class="metric-sub" id="kpiAnomRate">-</span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Possível GD Clandestina</span>
        <span class="metric-value" style="color: var(--solar)" id="kpiSolar">-</span>
        <span class="metric-sub">Fluxo Reverso s/ Cadastro</span>
      </div>
      <div class="metric-card">
        <span class="metric-label">Quarentena Telecom</span>
        <span class="metric-value" style="color: var(--text-dim)" id="kpiTelecom">-</span>
        <span class="metric-sub">Pacotes Mesh Degradados</span>
      </div>
    </div>

    <!-- Filter Bar -->
    <div class="filter-panel">
      <div class="search-box">
        <span class="search-icon">🔍</span>
        <input type="text" id="searchInput" class="search-input" placeholder="Buscar por UC, NIO, Alimentador ou Bairro..." oninput="onSearch()">
      </div>
      <select id="categorySelect" class="select-filter" onchange="onFilterChange()">
        <option value="">Todas as Categorias Diagnósticas</option>
        <option value="ANOMALIA_ELETRICA_SUSPEITA_FRAUDE_PNT">🚨 5. Anomalia Elétrica Pura / Suspeita Fraude PNT (PyOD ML)</option>
        <option value="POSSIVEL_GD_NAO_CADASTRADA">☀️ 6. Possível GD Não Cadastrada / Fluxo Reverso</option>
        <option value="FALHA_METROLOGICA_MEDIDOR_TRAVADO">⚠️ 7. Falha Metrológica / Medidor Travado (Tensão Normal)</option>
        <option value="MUDANCA_CLASSE_COMPORTAMENTAL">🔄 4. Mudança de Classe Comportamental no Tempo</option>
        <option value="INICIACAO_PERFIL_UC_NOVA">🆕 1. Iniciação de Perfil (UC Nova / 1ª Instalação <= 30d)</option>
        <option value="TRANSICAO_TROCA_MEDIDOR">🔧 2. Transição por Troca de Medidor (UC Existente)</option>
        <option value="ANOMALIA_COMUNICACAO_ISOLADA">📡 3A. Anomalia de Comunicação Isolada (Vizinhos Saudáveis >= 80%)</option>
        <option value="GARGALO_COLETIVO_RF_MESH">📶 3B. Gargalo Coletivo RF Mesh (Vizinhança Degradada)</option>
        <option value="OPERACAO_REGULAR">✔ Operação Regular</option>
      </select>
      <select id="feederSelect" class="select-filter" onchange="onFilterChange()">
        <option value="">Todos os Alimentadores</option>
      </select>
    </div>

    <!-- Table Card -->
    <div class="table-card">
      <div class="table-responsive">
        <table>
          <thead>
            <tr>
              <th>UC</th>
              <th>Alimentador</th>
              <th>Score Anomalia</th>
              <th>Diagnóstico Causal</th>
              <th>Consumo Mediano</th>
              <th>MAD Dispersão</th>
              <th>Desb. Corrente</th>
              <th>Reversão Ativa (RA)</th>
              <th>Ações</th>
            </tr>
          </thead>
          <tbody id="tableBody">
            <tr><td colspan="9" style="text-align:center; padding: 24px; color: var(--text-dim)">Carregando dados analíticos...</td></tr>
          </tbody>
        </table>
      </div>
      <div class="pagination">
        <span id="pageInfo">Exibindo 0 de 0 UCs</span>
        <div>
          <button class="btn btn-secondary" onclick="prevPage()" id="prevBtn">Anterior</button>
          <button class="btn btn-secondary" onclick="nextPage()" id="nextBtn">Próxima</button>
        </div>
      </div>
    </div>
  </div>

  <!-- Detail Drawer -->
  <div class="drawer-overlay" id="drawerOverlay" onclick="closeDrawer()"></div>
  <div class="drawer" id="detailDrawer">
    <div class="drawer-header">
      <div>
        <h3 id="drawerUc">UC 00000000</h3>
        <span id="drawerCat" class="badge">CATEGORIA</span>
      </div>
      <button class="drawer-close" onclick="closeDrawer()">&times;</button>
    </div>
    <div id="drawerContent"></div>
  </div>

  <script>
    let allData = [];
    let filteredData = [];
    let currentPage = 1;
    const pageSize = 50;

    async function fetchData() {
      try {
        const res = await fetch('/api/v1/anomalies');
        const json = await res.json();
        allData = json.data || [];
        populateFeeders(allData);
        updateKPIs(json.summary || {});
        applyFilters();
      } catch (err) {
        console.error(err);
      }
    }

    function populateFeeders(data) {
      const feeders = [...new Set(data.map(d => d.meta__feeder || d.ALIMENTADOR).filter(Boolean))].sort();
      const sel = document.getElementById('feederSelect');
      sel.innerHTML = '<option value="">Todos os Alimentadores</option>';
      feeders.forEach(f => {
        sel.innerHTML += `<option value="${f}">${f}</option>`;
      });
    }

    function updateKPIs(sum) {
      document.getElementById('kpiTotal').innerText = (sum.total_ucs || allData.length).toLocaleString();
      document.getElementById('kpiAnom').innerText = (sum.anomalies_detected || allData.filter(d => d.IS_ANOMALY).length).toLocaleString();
      const rate = allData.length > 0 ? (allData.filter(d => d.IS_ANOMALY).length / allData.length * 100).toFixed(2) : 0;
      document.getElementById('kpiAnomRate').innerText = `${rate}% da população`;
      document.getElementById('kpiSolar').innerText = (allData.filter(d => d.CATEGORIA_DIAGNOSTICA === 'POSSIVEL_GD_NAO_CADASTRADA').length).toLocaleString();
      document.getElementById('kpiTelecom').innerText = (allData.filter(d => d.CATEGORIA_DIAGNOSTICA === 'TELECOM_QUARANTINE').length).toLocaleString();
    }

    function applyFilters() {
      const q = (document.getElementById('searchInput').value || '').toLowerCase();
      const cat = document.getElementById('categorySelect').value;
      const feeder = document.getElementById('feederSelect').value;

      filteredData = allData.filter(d => {
        const uc = String(d.id__uc_id || d.UC || '').toLowerCase();
        const f = String(d.meta__feeder || d.ALIMENTADOR || '').toLowerCase();
        const matchQ = !q || uc.includes(q) || f.includes(q);
        const matchCat = !cat || d.CATEGORIA_DIAGNOSTICA === cat;
        const matchF = !feeder || (d.meta__feeder === feeder || d.ALIMENTADOR === feeder);
        return matchQ && matchCat && matchF;
      });

      currentPage = 1;
      renderTable();
    }

    function renderTable() {
      const tbody = document.getElementById('tableBody');
      const start = (currentPage - 1) * pageSize;
      const pageSlice = filteredData.slice(start, start + pageSize);

      if (pageSlice.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align:center; padding: 24px; color: var(--text-dim)">Nenhuma UC encontrada com os filtros selecionados.</td></tr>';
        document.getElementById('pageInfo').innerText = '0 de 0 UCs';
        return;
      }

      tbody.innerHTML = pageSlice.map(d => {
        const uc = d.id__uc_id || d.UC || '-';
        const feeder = d.meta__feeder || d.ALIMENTADOR || '-';
        const score = Number(d.ANOMALY_SCORE || d.FINAL_MOE_SCORE || 0);
        const scorePct = Math.round(score * 100);
        const scoreColor = score >= 0.75 ? 'var(--danger)' : (score >= 0.5 ? 'var(--warning)' : 'var(--primary)');
        const cat = d.CATEGORIA_DIAGNOSTICA || 'REGULAR';
        const badgeClass = getBadgeClass(cat);
        const faMed = Number(d.x__hist_fa_sum_median || 0).toFixed(2);
        const faMad = Number(d.x__hist_fa_sum_mad || 0).toFixed(2);
        const cImb = Number(d.x__hist_current_imbalance_max || 0).toFixed(2);
        const ra = Number(d.x__hist_ra_reversal_ratio_max || 0).toFixed(2);

        return `
          <tr>
            <td><strong>${uc}</strong></td>
            <td>${feeder}</td>
            <td>
              <div class="score-bar-bg">
                <div class="score-bar-fill" style="width: ${scorePct}%; background-color: ${scoreColor}"></div>
              </div>
              <span>${score.toFixed(4)}</span>
            </td>
            <td><span class="badge ${badgeClass}">${formatCat(cat)}</span></td>
            <td>${faMed} kWh</td>
            <td>${faMad}</td>
            <td>${cImb}</td>
            <td>${ra > 0 ? `<span style="color: var(--solar); font-weight: bold">${ra}</span>` : '0.00'}</td>
            <td>
              <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 11px;" onclick="openDetail('${uc}')">Inspecionar</button>
            </td>
          </tr>
        `;
      }).join('');

      document.getElementById('pageInfo').innerText = `Exibindo ${start + 1} a ${Math.min(start + pageSize, filteredData.length)} de ${filteredData.length.toLocaleString()} UCs`;
      document.getElementById('prevBtn').disabled = currentPage === 1;
      document.getElementById('nextBtn').disabled = start + pageSize >= filteredData.length;
    }

    function getBadgeClass(cat) {
      if (cat.includes('GD')) return 'badge-solar';
      if (cat.includes('BYPASS') || cat.includes('DESEQUILIBRIO')) return 'badge-fraud';
      if (cat.includes('TRAVADO') || cat.includes('ZERO')) return 'badge-meter';
      if (cat.includes('TELECOM')) return 'badge-telecom';
      if (cat === 'REGULAR') return 'badge-regular';
      return 'badge-tamper';
    }

    function formatCat(cat) {
      return cat.replace(/_/g, ' ');
    }

    function onSearch() { applyFilters(); }
    function onFilterChange() { applyFilters(); }
    function prevPage() { if (currentPage > 1) { currentPage--; renderTable(); } }
    function nextPage() { if ((currentPage * pageSize) < filteredData.length) { currentPage++; renderTable(); } }

    function openDetail(uc) {
      const d = allData.find(x => (x.id__uc_id === uc || x.UC === uc));
      if (!d) return;
      document.getElementById('drawerUc').innerText = `UC ${uc}`;
      const catEl = document.getElementById('drawerCat');
      catEl.innerText = formatCat(d.CATEGORIA_DIAGNOSTICA || 'REGULAR');
      catEl.className = `badge ${getBadgeClass(d.CATEGORIA_DIAGNOSTICA || '')}`;

      const content = `
        <div class="detail-section">
          <div class="detail-title">Contexto Elétrico e Cadastral</div>
          <div class="kv-grid">
            <div class="kv-box"><div class="kv-k">Alimentador</div><div class="kv-v">${d.meta__feeder || d.ALIMENTADOR || '-'}</div></div>
            <div class="kv-box"><div class="kv-k">Posto Operacional (Trafo)</div><div class="kv-v">${d.meta__posto || d.POSTO_OPERACIONAL || '-'}</div></div>
            <div class="kv-box"><div class="kv-k">Tensão Base</div><div class="kv-v">${d.x__tensao_base || 127} V</div></div>
            <div class="kv-box"><div class="kv-k">Faseamento</div><div class="kv-v">${d.x__fase_circuito || 'MONO'}</div></div>
          </div>
        </div>

        <div class="detail-section">
          <div class="detail-title">Métricas Comportamentais (30 Dias)</div>
          <div class="kv-grid">
            <div class="kv-box"><div class="kv-k">Consumo Diário Mediano</div><div class="kv-v">${Number(d.x__hist_fa_sum_median || 0).toFixed(2)} kWh</div></div>
            <div class="kv-box"><div class="kv-k">Dispersão Robusta (MAD)</div><div class="kv-v">${Number(d.x__hist_fa_sum_mad || 0).toFixed(2)}</div></div>
            <div class="kv-box"><div class="kv-k">Inclinação Tendência (Slope)</div><div class="kv-v">${Number(d.x__trend_fa_slope || 0).toFixed(4)}</div></div>
            <div class="kv-box"><div class="kv-k">Fator de Carga Médio</div><div class="kv-v">${Number(d.x__hist_load_factor_mean || 0).toFixed(2)}</div></div>
            <div class="kv-box"><div class="kv-k">Desbalanço de Corrente (P90)</div><div class="kv-v">${Number(d.x__hist_current_imbalance_max || 0).toFixed(2)}</div></div>
            <div class="kv-box"><div class="kv-k">Desbalanço Tensão (P95)</div><div class="kv-v">${Number(d.x__hist_voltage_imbalance_max || 0).toFixed(2)}%</div></div>
            <div class="kv-box"><div class="kv-k">Reversão Ativa (RA Max)</div><div class="kv-v">${Number(d.x__hist_ra_reversal_ratio_max || 0).toFixed(2)}</div></div>
            <div class="kv-box"><div class="kv-k">Dias de Reversão</div><div class="kv-v">${d.x__hist_ra_reversal_days || 0} dias</div></div>
          </div>
        </div>

        <div class="detail-section">
          <div class="detail-title">Integridade, Vizinhança Espacial & Ciclo de Vida</div>
          <div class="kv-grid">
            <div class="kv-box"><div class="kv-k">Cobertura da UC</div><div class="kv-v">${(Number(d.x__avg_interval_coverage || 0) * 100).toFixed(1)}%</div></div>
            <div class="kv-box"><div class="kv-k">Cobertura Mediana dos Vizinhos (Trafo/Geo)</div><div class="kv-v">${(Number(d.meta__peer_median_coverage || 0.95) * 100).toFixed(1)}%</div></div>
            <div class="kv-box"><div class="kv-k">Déficit vs. Vizinhos</div><div class="kv-v">${(Number(d.meta__coverage_deficit_vs_peers || 0) * 100).toFixed(1)}%</div></div>
            <div class="kv-box"><div class="kv-k">Dias Confiáveis (>70%)</div><div class="kv-v">${d.x__coverage_reliable_days || 0} dias</div></div>
            <div class="kv-box"><div class="kv-k">Medidores na História da UC</div><div class="kv-v">${d.meta__n_meters_historical || 1} medidor(es)</div></div>
            <div class="kv-box"><div class="kv-k">Idade do Medidor Atual</div><div class="kv-v">${d.x__meter_age_days || 0} dias</div></div>
            <div class="kv-box"><div class="kv-k">Classe Comportamental</div><div class="kv-v">${d.CLASSE_COMPORTAMENTAL || '-'}</div></div>
            <div class="kv-box"><div class="kv-k">Método de Detecção</div><div class="kv-v">${d.METODO_DETECCAO_APLICADO || '-'}</div></div>
          </div>
        </div>
        <div class="detail-section">
          <div class="detail-title">Trajetória Longitudinal de Classes entre Cutoffs</div>
          <div class="kv-box"><div class="kv-k">Evolução Temporal</div><div class="kv-v" style="white-space:normal">${d.TRAJETORIA_CLASSES_TEMPORAL || d.CLASSE_COMPORTAMENTAL || '-'}</div></div>
        </div>
      `;
      document.getElementById('drawerContent').innerHTML = content;
      document.getElementById('detailDrawer').classList.add('open');
      document.getElementById('drawerOverlay').classList.add('open');
    }

    function closeDrawer() {
      document.getElementById('detailDrawer').classList.remove('open');
      document.getElementById('drawerOverlay').classList.remove('open');
    }

    function exportData(fmt) {
      window.location.href = '/api/v1/export?format=' + fmt;
    }

    window.onload = fetchData;
  </script>
</body>
</html>
"""


class CockpitHandler(BaseHTTPRequestHandler):
    """Zero-dependency HTTP Handler for Cockpit Web and REST endpoints."""

    def log_message(self, format, *args):
        try:
            super().log_message(format, *args)
        except Exception:
            pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        path = parsed.path
        query = urllib.parse.parse_qs(parsed.query)

        # 1. Static HTML
        if path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_TEMPLATE.encode("utf-8"))
            return

        # 2. API: Anomalies & Classified List
        if path == "/api/v1/anomalies":
            df = _load_data()
            summary = _load_summary()
            if df.is_empty():
                payload = {"data": [], "summary": summary}
            else:
                # Select essential columns to keep payload fast and light
                view_cols = [
                    c for c in [
                        "id__uc_id", "UC", "meta__cutoff_date", "meta__feeder", "ALIMENTADOR", "meta__posto", "POSTO_OPERACIONAL",
                        "ANOMALY_SCORE", "COPOD_SCORE", "ECOD_SCORE", "FINAL_MOE_SCORE", "IS_ANOMALY",
                        "CATEGORIA_DIAGNOSTICA", "SUBTIPO_DIAGNOSTICO", "METODO_DETECCAO_APLICADO",
                        "CLASSE_COMPORTAMENTAL", "MUDOU_CLASSE_TEMPORAL", "TRAJETORIA_CLASSES_TEMPORAL",
                        "meta__n_meters_historical", "meta__peer_median_coverage", "meta__coverage_deficit_vs_peers",
                        "x__hist_fa_sum_median", "x__hist_fa_sum_mad", "x__trend_fa_slope",
                        "x__hist_current_imbalance_max", "x__hist_voltage_imbalance_max",
                        "x__hist_ra_reversal_ratio_max", "x__hist_ra_reversal_days",
                        "x__avg_interval_coverage", "x__coverage_reliable_days", "x__alarm_count_window",
                        "x__meter_age_days", "x__tensao_base", "x__fase_circuito", "x__hist_load_factor_mean"
                    ]
                    if c in df.columns
                ]
                records = df.select(view_cols).to_dicts()
                payload = {"data": records, "summary": summary}

            self._send_json(payload)
            return

        # 3. API: Export Data
        if path == "/api/v1/export":
            if CLASSIFIED_CSV.exists():
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", "attachment; filename=classified_ucs_latest.csv")
                self.end_headers()
                with open(CLASSIFIED_CSV, "rb") as f:
                    self.wfile.write(f.read())
            elif CLASSIFIED_PARQUET.exists():
                df = pl.read_parquet(CLASSIFIED_PARQUET)
                self.send_response(200)
                self.send_header("Content-Type", "text/csv; charset=utf-8")
                self.send_header("Content-Disposition", "attachment; filename=classified_ucs_latest.csv")
                self.end_headers()
                self.wfile.write(df.write_csv(separator=";").encode("utf-8"))
            else:
                self.send_error(404, "Arquivo de exportação não encontrado.")
            return

        # 4. API: Health Check
        if path == "/api/v1/health":
            self._send_json({"status": "OK", "has_data": CLASSIFIED_PARQUET.exists()})
            return

        self.send_error(404, "Not Found")

    def _send_json(self, data: Any):
        self.send_response(200)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))


def run_web_cockpit(host: str = "127.0.0.1", port: int = 8050, open_browser: bool = True):
    """Starts the zero-dependency web cockpit server."""
    server_address = (host, port)
    httpd = HTTPServer(server_address, CockpitHandler)
    url = f"http://localhost:{port}"

    print(f"\n⚡ Cockpit Analítico AMI rodando em: {url}")
    print("Pressione Ctrl+C para encerrar o servidor.\n")

    if open_browser:
        webbrowser.open(url)

    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nEncerrando o servidor web...")
        httpd.server_close()


def main():
    parser = argparse.ArgumentParser(description="Run Cockpit Analítico AMI Web Server.")
    parser.add_argument("--port", type=int, default=8050, help="HTTP server port.")
    parser.add_argument("--no-browser", action="store_true", help="Do not open browser automatically.")
    args = parser.parse_args()

    run_web_cockpit(port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()

