"""Experimental Contracts, Fail-Closed Invariant Validators, Provenance Manifests & KPI Registry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date, datetime
import hashlib
import json
from pathlib import Path
import platform
import subprocess
from typing import Any

import numpy as np
import polars as pl
import sklearn


VALID_MANIFEST_STATES: tuple[str, ...] = (
    "created",
    "running",
    "completed",
    "validated",
    "invalid",
    "failed",
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _parse_scalar_yaml(val: str) -> Any:
    """Parse a scalar YAML value safely without external dependencies."""
    s = val.strip()
    if s in ("null", "Null", "NULL", "~", ""):
        return None
    if s in ("true", "True", "TRUE"):
        return True
    if s in ("false", "False", "FALSE"):
        return False
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar_yaml(item) for item in inner.split(",")]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return s


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Dependency-free nested mapping YAML loader (supports optional pyyaml fallback)."""
    try:
        import yaml  # type: ignore[import-untyped]
        loaded = yaml.safe_load(text)
        if isinstance(loaded, dict):
            return loaded
    except ImportError:
        pass

    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]

    for raw_line in text.splitlines():
        # Strip inline comments outside quotes
        line = raw_line.split("#", 1)[0].rstrip() if "#" not in raw_line.split(":", 1)[-1] or raw_line.lstrip().startswith("#") else raw_line.rstrip()
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        stripped = line.strip()
        if ":" not in stripped:
            continue
        key, rest = stripped.split(":", 1)
        key = key.strip()
        rest = rest.strip()

        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]

        if rest == "":
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            # Strip trailing comment if any
            if " #" in rest and not (rest.startswith('"') or rest.startswith("'")):
                rest = rest.split(" #", 1)[0].strip()
            parent[key] = _parse_scalar_yaml(rest)
    return root


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file (streaming)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(65536)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(PROJECT_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=3,
        ).strip()
        return out or "unknown"
    except Exception:
        head_file = PROJECT_ROOT / ".git" / "HEAD"
        if head_file.exists():
            return head_file.read_text(encoding="utf-8").strip()[:40]
        return "unavailable"


def atomic_write_text(path: Path, text: str) -> None:
    """Write text atomically using a temp sibling file and os.replace semantics."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def atomic_write_json(path: Path, payload: Any) -> None:
    atomic_write_text(path, json.dumps(payload, indent=2, default=str, ensure_ascii=False))


@dataclass(frozen=True)
class ExperimentConfig:
    """Validated experiment configuration contract."""

    experiment_id: str
    data_source: str
    scope: str
    start_date: str
    end_date: str
    timezone: str
    unit_of_analysis: str
    pipeline_id: str
    train: dict[str, Any] = field(default_factory=dict)
    inference: dict[str, Any] = field(default_factory=dict)
    quality: dict[str, Any] = field(default_factory=dict)
    execution: dict[str, Any] = field(default_factory=dict)
    raw_yaml: str = ""

    @property
    def start_dt(self) -> date:
        return date.fromisoformat(str(self.start_date)[:10])

    @property
    def end_dt(self) -> date:
        return date.fromisoformat(str(self.end_date)[:10])

    @property
    def train_end_dt(self) -> date | None:
        v = self.train.get("end_date")
        return date.fromisoformat(str(v)[:10]) if v else None

    @property
    def seed(self) -> int:
        return int(self.execution.get("seed", 42))

    @property
    def coverage_threshold(self) -> float:
        return float(self.quality.get("coverage_threshold", 0.70))

    @property
    def min_window_coverage(self) -> float:
        return float(self.quality.get("min_window_coverage", 0.60))

    @property
    def min_reliable_days(self) -> int:
        return int(self.quality.get("min_reliable_days", 15))

    @property
    def normalization_mode(self) -> str:
        return str(self.inference.get("normalization_mode", "frozen_quantile"))

    @property
    def freeze_threshold(self) -> bool:
        return bool(self.inference.get("frozen_thresholds", True))

    def validate(self, *, require_train_cutoff: bool = False) -> None:
        """Fail-closed validation of configuration fields before launching any run."""
        if not self.experiment_id or not self.pipeline_id:
            raise ValueError("experiment_id and pipeline_id are mandatory in config.")
        if self.start_dt > self.end_dt:
            raise ValueError(f"start_date ({self.start_dt}) cannot be after end_date ({self.end_dt}).")
        if not (0.0 <= self.coverage_threshold <= 1.0):
            raise ValueError(f"coverage_threshold must be in [0, 1], got {self.coverage_threshold}")
        if self.pipeline_id == "oot_multiview" or require_train_cutoff:
            if self.train_end_dt is None:
                raise ValueError("OOT experiments require explicit train.end_date before eval start_date.")
            if self.train_end_dt >= self.start_dt:
                raise ValueError(
                    f"Temporal leakage violation: train.end_date ({self.train_end_dt}) >= eval start_date ({self.start_dt})."
                )


def load_experiment_config(
    config_path: Path | str,
    *,
    pipeline_override: str | None = None,
) -> ExperimentConfig:
    """Load and validate an ExperimentConfig from a YAML file."""
    p = Path(config_path)
    if not p.exists():
        raise FileNotFoundError(f"Experiment config not found: {p}")
    raw_text = p.read_text(encoding="utf-8")
    data = _simple_yaml_load(raw_text)

    cfg = ExperimentConfig(
        experiment_id=str(data.get("experiment_id", "")),
        data_source=str(data.get("data_source", "local_lake")),
        scope=str(data.get("scope", "study_area")),
        start_date=str(data.get("start_date", "")),
        end_date=str(data.get("end_date", "")),
        timezone=str(data.get("timezone", "America/Sao_Paulo")),
        unit_of_analysis=str(data.get("unit_of_analysis", "uc_cutoff")),
        pipeline_id=str(pipeline_override or data.get("pipeline_id", "")),
        train=dict(data.get("train") or {}),
        inference=dict(data.get("inference") or {}),
        quality=dict(data.get("quality") or {}),
        execution=dict(data.get("execution") or {}),
        raw_yaml=raw_text,
    )
    cfg.validate(require_train_cutoff=(cfg.pipeline_id == "oot_multiview"))
    return cfg


@dataclass(frozen=True)
class KPIContract:
    """Formal mathematical and semantic definition of an indicator (Section 3.2)."""

    metric_id: str
    display_name: str
    description: str
    grain: str
    numerator: str
    denominator: str
    source: str
    filters: str
    deduplication: str
    time_reference: str
    null_policy: str
    validation: str


KPI_REGISTRY: tuple[KPIContract, ...] = (
    KPIContract(
        metric_id="evaluated_windows",
        display_name="Janelas Avaliadas (UC × Cutoff)",
        description="Total de janelas retrospectivas avaliadas na série temporal.",
        grain="UC × cutoff_date",
        numerator="COUNT(*) em features de janela",
        denominator="N/A (contagem absoluta)",
        source="uc_window_features / daily_cuts",
        filters="REPORT_DAY in (cutoff - 30d, cutoff]",
        deduplication="1 linha por (UC, cutoff_date)",
        time_reference="cutoff_date",
        null_policy="Sem descarte silencioso",
        validation="N_total == N_eligible + N_quarantine + N_other",
    ),
    KPIContract(
        metric_id="distinct_ucs",
        display_name="UCs Únicas",
        description="Unidades consumidoras distintas na coorte ou série.",
        grain="UC (id__uc_id normalizado)",
        numerator="COUNT(DISTINCT id__uc_id)",
        denominator="População ativa elegível ou total do cadastro",
        source="decisions.parquet / population.parquet",
        filters="Por corte ou dedup no último corte",
        deduplication="keep='last' ordenado por cutoff_date",
        time_reference="cutoff_date",
        null_policy="IDs vazios rejeitados",
        validation="distinct_ucs <= evaluated_windows",
    ),
    KPIContract(
        metric_id="eligible_ucs",
        display_name="UCs Elegíveis (Telemetria Limpa)",
        description="UCs com completude e dias confiáveis suficientes para análise elétrica.",
        grain="UC × cutoff_date",
        numerator="COUNT(UC WHERE is_eligible_clean == True)",
        denominator="N_total no corte",
        source="quality.parquet",
        filters="reliable_days >= min_reliable_days AND avg_coverage >= min_window_coverage",
        deduplication="1 linha por (UC, cutoff_date)",
        time_reference="cutoff_date",
        null_policy="Cobertura ausente = 0.0 -> quarentena",
        validation="E_t ∩ Q_t == ∅",
    ),
    KPIContract(
        metric_id="quarantine_ucs",
        display_name="UCs em Quarentena Telecom",
        description="UCs retidas por falha de comunicação ou cobertura insuficiente.",
        grain="UC × cutoff_date",
        numerator="COUNT(UC WHERE telecom_status LIKE 'QUARANTINE%')",
        denominator="N_total no corte",
        source="quality.parquet",
        filters="NOT is_eligible_clean AND NOT is_other",
        deduplication="1 linha por (UC, cutoff_date)",
        time_reference="cutoff_date",
        null_policy="Excluídas do denominador de anomalias elétricas",
        validation="E_t + Q_t + O_t == N_total",
    ),
    KPIContract(
        metric_id="statistical_anomaly_ucs",
        display_name="UCs com Anomalia Estatística",
        description="UCs elegíveis que excedem o limiar congelado do modelo estatístico.",
        grain="UC × cutoff_date",
        numerator="COUNT(UC WHERE is_eligible_clean AND is_statistical_anomaly)",
        denominator="N_eligible (apenas elegíveis)",
        source="decisions.parquet",
        filters="is_eligible_clean == True AND score >= frozen_threshold",
        deduplication="1 linha por (UC, cutoff_date)",
        time_reference="cutoff_date",
        null_policy="Quarentena excluída do numerador e denominador",
        validation="0 <= statistical_anomaly_ucs <= eligible_ucs",
    ),
    KPIContract(
        metric_id="diagnostic_hypothesis_ucs",
        display_name="UCs com Hipótese Diagnóstica",
        description="UCs com hipótese elétrica ou metrológica atribuída sem confirmação causal.",
        grain="UC × cutoff_date",
        numerator="COUNT(UC WHERE categoria_diagnostica IN actionable_hypotheses)",
        denominator="N_eligible no corte",
        source="decisions.parquet",
        filters="is_eligible_clean == True",
        deduplication="1 linha por (UC, cutoff_date)",
        time_reference="cutoff_date",
        null_policy="Nunca usar termo 'confirmada'",
        validation="Mutualmente exclusivo por categoria primária",
    ),
    KPIContract(
        metric_id="candidate_ucs",
        display_name="Candidatos à Vistoria (Campo)",
        description="UCs aprovadas pela política operacional de encaminhamento.",
        grain="UC × cutoff_date",
        numerator="COUNT(UC WHERE is_candidato_vistoria == True)",
        denominator="N_eligible no corte",
        source="candidate_sets.parquet",
        filters="is_eligible_clean AND hypothesis_actionable AND materiality AND persistence",
        deduplication="Conjunto único C_t por corte",
        time_reference="cutoff_date",
        null_policy="N/A",
        validation="C_t == I_t ∪ P_t; C_{t-1} == O_t ∪ P_t",
    ),
    KPIContract(
        metric_id="queue_entries",
        display_name="Entradas na Fila (I_t)",
        description="UCs que ingressam no conjunto de candidatos no corte t.",
        grain="UC × cutoff_date",
        numerator="|C_t \\ C_{t-1}|",
        denominator="N/A ou |C_{t-1}| para taxa de renovação",
        source="candidate_sets.parquet",
        filters="t > t_0",
        deduplication="Conjunto matemático estrito",
        time_reference="cutoff_date",
        null_policy="No corte t_0, I_0 = C_0 e O_0 = ∅",
        validation="I_t ∩ P_t == ∅",
    ),
    KPIContract(
        metric_id="queue_exits",
        display_name="Saídas da Fila (O_t)",
        description="UCs presentes em t-1 que deixaram a fila em t.",
        grain="UC × cutoff_date",
        numerator="|C_{t-1} \\ C_t|",
        denominator="|C_{t-1}| para taxa de saída",
        source="candidate_sets.parquet",
        filters="t > t_0",
        deduplication="Conjunto matemático estrito",
        time_reference="cutoff_date",
        null_policy="No corte t_0, O_0 = ∅",
        validation="O_t ∩ P_t == ∅",
    ),
    KPIContract(
        metric_id="queue_persistence",
        display_name="Permanência na Fila (P_t)",
        description="UCs mantidas consecutivamente entre t-1 e t.",
        grain="UC × cutoff_date",
        numerator="|C_t ∩ C_{t-1}|",
        denominator="|C_{t-1}| para taxa de retenção",
        source="candidate_sets.parquet",
        filters="t > t_0",
        deduplication="Conjunto matemático estrito",
        time_reference="cutoff_date",
        null_policy="No corte t_0, P_0 = ∅",
        validation="C_t == I_t ∪ P_t AND C_{t-1} == O_t ∪ P_t",
    ),
)


@dataclass
class InvariantCheckResult:
    name: str
    passed: bool
    details: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


@dataclass
class InvariantReport:
    """Aggregated fail-closed invariant verification result."""

    checks: list[InvariantCheckResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return bool(self.checks) and all(c.passed for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "invariants_verified": self.all_passed,
            "total_checks": len(self.checks),
            "passed_checks": sum(1 for c in self.checks if c.passed),
            "failed_checks": sum(1 for c in self.checks if not c.passed),
            "checks": [asdict(c) for c in self.checks],
        }


def validate_population_partition(partitions: list[dict[str, Any]]) -> InvariantCheckResult:
    """Verify exact mutually-exclusive population partitioning per cutoff:
    N_total == N_eligible + N_quarantine + N_other and E ∩ Q == Q ∩ O == E ∩ O == ∅.
    """
    errors: list[str] = []
    checked = 0
    for row in partitions:
        checked += 1
        cutoff = str(row.get("cutoff_date", f"idx_{checked}"))
        e_set = set(row.get("eligible_ucs", set()))
        q_set = set(row.get("quarantine_ucs", set()))
        o_set = set(row.get("other_ucs", set()))
        n_total = int(row.get("total_count", len(e_set) + len(q_set) + len(o_set)))
        n_elig = int(row.get("eligible_count", len(e_set)))
        n_quar = int(row.get("quarantine_count", len(q_set)))
        n_other = int(row.get("other_count", len(o_set)))

        if n_total != n_elig + n_quar + n_other:
            errors.append(
                f"[{cutoff}] Partition count mismatch: total={n_total} != eligible({n_elig}) + quarantine({n_quar}) + other({n_other})"
            )

        if e_set or q_set or o_set:
            if len(e_set & q_set) > 0:
                errors.append(f"[{cutoff}] Overlap eligible ∩ quarantine: {len(e_set & q_set)} UCs")
            if len(e_set & o_set) > 0:
                errors.append(f"[{cutoff}] Overlap eligible ∩ other: {len(e_set & o_set)} UCs")
            if len(q_set & o_set) > 0:
                errors.append(f"[{cutoff}] Overlap quarantine ∩ other: {len(q_set & o_set)} UCs")
            if len(e_set) != n_elig or len(q_set) != n_quar or len(o_set) != n_other:
                errors.append(
                    f"[{cutoff}] Unique UC set size mismatch vs reported counts: "
                    f"sets=({len(e_set)}, {len(q_set)}, {len(o_set)}) vs counts=({n_elig}, {n_quar}, {n_other})"
                )

    return InvariantCheckResult(
        name="population_partition_conservation",
        passed=(len(errors) == 0 and checked > 0),
        details={"cutoffs_checked": checked},
        errors=errors[:25],
    )


def validate_portfolio_queue_sets(
    candidates_by_date: dict[str, set[str]],
    reported_rows: list[dict[str, Any]] | None = None,
) -> InvariantCheckResult:
    """Verify exact set-level candidate queue dynamics across consecutive cutoffs:
    I_t = C_t \\ C_{t-1}, O_t = C_{t-1} \\ C_t, P_t = C_t ∩ C_{t-1}
    asserting C_t == I_t ∪ P_t and C_{t-1} == O_t ∪ P_t by UC identity.
    """
    errors: list[str] = []
    ordered_dates = sorted(candidates_by_date.keys())
    if not ordered_dates:
        return InvariantCheckResult(
            name="portfolio_queue_set_identities",
            passed=False,
            details={"cutoffs_checked": 0},
            errors=["No candidate cutoffs provided for queue invariant verification."],
        )

    rep_by_dt = {str(r["CUTOFF_DATE"]): r for r in reported_rows} if reported_rows else {}
    prev_c: set[str] | None = None

    for dt_str in ordered_dates:
        curr_c = set(candidates_by_date[dt_str])
        if prev_c is None:
            entries = set(curr_c)
            exits: set[str] = set()
            persisted: set[str] = set()
        else:
            entries = curr_c - prev_c
            exits = prev_c - curr_c
            persisted = curr_c & prev_c

            if (entries | persisted) != curr_c:
                errors.append(f"[{dt_str}] Set identity failed: (I_t ∪ P_t) != C_t")
            if (exits | persisted) != prev_c:
                errors.append(f"[{dt_str}] Set identity failed: (O_t ∪ P_t) != C_{{t-1}}")
            if len(entries & persisted) > 0:
                errors.append(f"[{dt_str}] Disjointness failed: I_t ∩ P_t != ∅")
            if len(exits & persisted) > 0:
                errors.append(f"[{dt_str}] Disjointness failed: O_t ∩ P_t != ∅")
            if len(curr_c) != len(prev_c) - len(exits) + len(entries):
                errors.append(f"[{dt_str}] Queue balance arithmetic mismatch")

        if dt_str in rep_by_dt:
            rep = rep_by_dt[dt_str]
            if int(rep.get("CANDIDATOS_VISTORIA", -1)) != len(curr_c):
                errors.append(f"[{dt_str}] Reported candidates {rep.get('CANDIDATOS_VISTORIA')} != |C_t|={len(curr_c)}")
            if int(rep.get("FILA_ENTRADAS", -1)) != len(entries):
                errors.append(f"[{dt_str}] Reported entries {rep.get('FILA_ENTRADAS')} != |I_t|={len(entries)}")
            if int(rep.get("FILA_SAIDAS", -1)) != len(exits):
                errors.append(f"[{dt_str}] Reported exits {rep.get('FILA_SAIDAS')} != |O_t|={len(exits)}")
            if int(rep.get("FILA_PERMANENCIA", -1)) != len(persisted):
                errors.append(f"[{dt_str}] Reported permanence {rep.get('FILA_PERMANENCIA')} != |P_t|={len(persisted)}")

        prev_c = curr_c

    return InvariantCheckResult(
        name="portfolio_queue_set_identities",
        passed=(len(errors) == 0),
        details={"cutoffs_checked": len(ordered_dates)},
        errors=errors[:25],
    )


def validate_temporal_no_leakage(
    *,
    train_max_date: date | None,
    eval_start_date: date,
    window_df: pl.DataFrame | None = None,
    cutoff_col: str = "CUTOFF_DATE",
    obs_date_col: str = "REPORT_DAY",
    max_allowed_date: date | None = None,
) -> InvariantCheckResult:
    """Verify zero temporal future leakage between training and evaluation and inside windows."""
    errors: list[str] = []
    if train_max_date is not None and train_max_date >= eval_start_date:
        errors.append(f"Train max date ({train_max_date}) >= eval start date ({eval_start_date}).")

    if window_df is not None and not window_df.is_empty():
        if cutoff_col in window_df.columns and obs_date_col in window_df.columns:
            leaks = window_df.filter(
                pl.col(obs_date_col).cast(pl.Date) > pl.col(cutoff_col).cast(pl.Date)
            )
            if leaks.height > 0:
                errors.append(f"Detected {leaks.height} window observations with {obs_date_col} > {cutoff_col}.")
        if max_allowed_date is not None:
            for col in (cutoff_col, obs_date_col):
                if col in window_df.columns:
                    future = window_df.filter(pl.col(col).cast(pl.Date) > max_allowed_date)
                    if future.height > 0:
                        errors.append(f"Detected {future.height} future rows in {col} > {max_allowed_date}.")

    return InvariantCheckResult(
        name="temporal_no_future_leakage",
        passed=(len(errors) == 0),
        details={
            "train_max_date": str(train_max_date) if train_max_date else None,
            "eval_start_date": str(eval_start_date),
        },
        errors=errors[:25],
    )


def validate_score_and_kpi_integrity(
    scores: np.ndarray | list[float],
    metrics_rows: list[dict[str, Any]] | None = None,
) -> InvariantCheckResult:
    """Verify score domain [0, 1], finiteness, and numerator <= denominator bounds."""
    errors: list[str] = []
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        errors.append("Score array is empty.")
    else:
        if np.any(np.isnan(arr)) or np.any(np.isinf(arr)):
            errors.append("Non-finite (NaN/Inf) score values detected.")
        if float(np.min(arr)) < -1e-9 or float(np.max(arr)) > 1.0 + 1e-9:
            errors.append(f"Scores outside [0, 1]: min={np.min(arr)}, max={np.max(arr)}")

    if metrics_rows:
        for r in metrics_rows:
            dt = str(r.get("CUTOFF_DATE", "?"))
            elig = int(r.get("ELEGIVEIS_LIMPOS", 0))
            cand = int(r.get("CANDIDATOS_VISTORIA", 0))
            anom = int(r.get("ANOMALIAS_DETECTADAS", 0))
            if cand > elig:
                errors.append(f"[{dt}] Candidates ({cand}) > Eligible UCs ({elig})")
            if anom > elig:
                errors.append(f"[{dt}] Statistical anomalies on eligible ({anom}) > Eligible UCs ({elig})")

    return InvariantCheckResult(
        name="score_bounds_and_kpi_denominators",
        passed=(len(errors) == 0),
        details={"n_scores": int(arr.size), "n_metric_rows": len(metrics_rows or [])},
        errors=errors[:25],
    )


class RunManifest:
    """Lifecycle-enforced provenance manifest for an experiment run."""

    def __init__(self, run_dir: Path, config: ExperimentConfig) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.run_dir / "manifest.json"
        self.invariants_path = self.run_dir / "invariants.json"

        self.data: dict[str, Any] = {
            "run_id": self.run_dir.name,
            "experiment_id": config.experiment_id,
            "pipeline_id": config.pipeline_id,
            "status": "created",
            "invariants_verified": False,
            "git_commit": get_git_commit(),
            "config_sha256": sha256_text(config.raw_yaml) if config.raw_yaml else sha256_text(json.dumps(asdict(config), sort_keys=True, default=str)),
            "config": {k: v for k, v in asdict(config).items() if k != "raw_yaml"},
            "environment": {
                "python": platform.python_version(),
                "os": platform.platform(),
                "polars": pl.__version__,
                "numpy": np.__version__,
                "sklearn": sklearn.__version__,
            },
            "input_hashes": {},
            "parameters": {},
            "artifacts": {},
            "errors": [],
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "completed_at": None,
        }
        self._save()

    def _save(self) -> None:
        self.data["updated_at"] = datetime.now().isoformat()
        atomic_write_json(self.manifest_path, self.data)

    def transition(self, new_state: str, *, error_msg: str | None = None) -> None:
        if new_state not in VALID_MANIFEST_STATES:
            raise ValueError(f"Invalid manifest state: {new_state}")
        if new_state == "validated" and not self.data.get("invariants_verified", False):
            raise RuntimeError("Fail-closed block: cannot transition manifest to 'validated' without verified invariants.")
        self.data["status"] = new_state
        if error_msg:
            self.data["errors"].append(error_msg)
        if new_state in ("completed", "validated", "invalid", "failed"):
            self.data["completed_at"] = datetime.now().isoformat()
        self._save()

    def register_input(self, name: str, path: Path) -> None:
        if path.exists() and path.is_file():
            self.data["input_hashes"][name] = {
                "path": str(path),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path) if path.stat().st_size < 100_000_000 else "skipped_gt_100mb",
            }
        else:
            self.data["input_hashes"][name] = {"path": str(path), "exists": path.exists()}
        self._save()

    def register_artifact(self, key: str, path: Path) -> None:
        self.data["artifacts"][key] = str(path)
        self._save()

    def record_parameters(self, params: dict[str, Any]) -> None:
        self.data["parameters"].update(params)
        self._save()

    def finalize_with_invariants(self, report: InvariantReport) -> bool:
        """Persist invariant report and atomically enforce validated vs invalid state."""
        atomic_write_json(self.invariants_path, report.to_dict())
        self.register_artifact("invariants", self.invariants_path)
        if report.all_passed:
            self.data["invariants_verified"] = True
            self.transition("validated")
            return True
        else:
            self.data["invariants_verified"] = False
            failed_msgs = [
                f"{c.name}: {'; '.join(c.errors[:3])}"
                for c in report.checks
                if not c.passed
            ]
            self.transition("invalid", error_msg=" | ".join(failed_msgs))
            return False
