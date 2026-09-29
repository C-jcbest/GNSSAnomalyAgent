import { useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import type { LineSeriesOption } from "echarts/charts";
import {
  Activity,
  BookOpenText,
  CalendarDays,
  Check,
  ChevronDown,
  ChevronRight,
  Crosshair,
  Database,
  Filter,
  Layers3,
  LoaderCircle,
  Play,
  RotateCcw,
  Search,
} from "lucide-react";
import DocsPage from "./DocsPage";

type Triple = [number, number, number];
type Status = "queued" | "running" | "complete" | "failed";
type CaseType = "normal" | "global_extremum" | "trend" | "mean_shift";
type GenerationType = CaseType | "all";
type CaseKind = CaseType;
type Axis = "N" | "E" | "U";
type EventTruth = {
  event_id: string;
  type: Exclude<CaseType, "normal">;
  axis: Axis;
  start_index: number;
  end_index: number;
  start_date: string;
  end_date: string;
  persistent: boolean;
  operation: "native_extremum" | "add";
  target_offset_mm: number;
  sigma_mm: number;
  observed_end_mm: number;
  source: string;
  native_parameters: Record<string, number | boolean | string>;
};
const caseTypeLabels: Record<CaseType, string> = {
  normal: "正常", global_extremum: "Global Extremum", trend: "Trend",
  mean_shift: "Transient Mean Shift",
};
const caseTypes = Object.keys(caseTypeLabels) as CaseType[];
const kindLabels = caseTypeLabels;
const generationTypeLabels: Record<GenerationType, string> = {
  all: "全部类型（同背景配对）", ...caseTypeLabels,
};
type ComponentKey = "noise" | "delta" | "observed";
const componentLabels: Record<ComponentKey, string> = {
  noise: "原始高斯噪声", delta: "异常净改变量", observed: "最终观测",
};
const componentColors: Record<ComponentKey, string> = {
  noise: "#db6b50", delta: "#147d72", observed: "#253b54",
};

type Manifest = {
  dataset_id: string;
  created_at: string;
  status: Status;
  generator_version: string;
  request: { seed: number; count: number; case_type: GenerationType };
  type_counts: Partial<Record<CaseKind, number>>;
  generated_cases: number;
  cases: {
    case_id: string;
    case_seed: number;
    case_type: CaseKind;
    event_count: number;
    background_group: string;
    group_axis: Axis;
    group_sign: -1 | 1;
  }[];
  error: string | null;
};
type CaseInput = {
  case_id: string;
  dates: string[];
  reference_coordinate_mm: Triple;
  observed_coordinate_mm: Triple[];
  displacement_mm: Triple[];
  horizontal_offset_mm: number[];
  spatial_offset_mm: number[];
};
type CaseTruth = {
  events: EventTruth[];
  measurement_noise_mm: Triple[];
  anomaly_delta_mm: Triple[];
  noise_seed: number;
  position_seed: number;
  tods_seed: number;
  background_group: string;
};

const axisNames = ["N", "E", "U"] as const;
const colors = { N: "#147d72", E: "#db6b50", U: "#ad8525", H: "#4b5f91" };
const statusLabels: Record<Status, string> = {
  queued: "排队中",
  running: "生成中",
  complete: "已完成",
  failed: "生成失败",
};

async function api<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(path, { cache: "no-store", ...options });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    throw new Error(typeof body.detail === "string" ? body.detail : Array.isArray(body.detail) ? body.detail.map((item: { msg: string }) => item.msg).join("；") : `请求失败 (${response.status})`);
  }
  return response.json();
}

function formattedDate(value: string) {
  return new Date(value).toLocaleString("zh-CN", {
    year: "numeric",
    month: "2-digit",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function valueLabel(value: number) {
  return `${value.toFixed(2)} mm`;
}

export default function App() {
  const [datasets, setDatasets] = useState<Manifest[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<Manifest | null>(null);
  const [caseId, setCaseId] = useState<string | null>(null);
  const [caseInput, setCaseInput] = useState<CaseInput | null>(null);
  const [caseTruth, setCaseTruth] = useState<CaseTruth | null>(null);
  const [seed, setSeed] = useState("20260929");
  const [count, setCount] = useState("24");
  const [caseType, setCaseType] = useState<GenerationType>("all");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [page, setCurrentPage] = useState<"datasets" | "runs" | "docs">(() => {
    const requested = new URLSearchParams(location.search).get("view");
    return requested === "docs" || requested === "runs" ? requested : "datasets";
  });
  function setPage(next: "datasets" | "runs" | "docs") {
    const url = new URL(location.href);
    url.searchParams.set("view", next);
    url.hash = "";
    if (next !== "docs") url.searchParams.delete("doc");
    history.pushState(null, "", url);
    setCurrentPage(next);
    window.scrollTo({ top: 0, behavior: "auto" });
  }
  useEffect(() => {
    const restore = () => {
      const requested = new URLSearchParams(location.search).get("view");
      setCurrentPage(requested === "docs" || requested === "runs" ? requested : "datasets");
    };
    window.addEventListener("popstate", restore);
    return () => window.removeEventListener("popstate", restore);
  }, []);
  const [view, setView] = useState<"observed" | "components">("observed");
  const [visible, setVisible] = useState({
    N: true,
    E: true,
    U: true,
    H: false,
  });
  const [componentAxis, setComponentAxis] = useState<0 | 1 | 2>(0);
  const [showTruth, setShowTruth] = useState(false);
  const [selectedEventId, setSelectedEventId] = useState<string | null>(null);
  const [eventAxisFilter, setEventAxisFilter] = useState<Axis | "all">("all");
  const [eventFamilyFilter, setEventFamilyFilter] = useState<"all" | "global_extremum" | "trend" | "mean_shift">("all");
  const [truthVisible, setTruthVisible] = useState(true);
  const [truthLoading, setTruthLoading] = useState(false);
  const [componentsVisible, setComponentsVisible] = useState<Record<ComponentKey, boolean>>({
    noise: true, delta: true, observed: true,
  });
  const [dateTarget, setDateTarget] = useState("");
  const [query, setQuery] = useState("");
  const [expandedGroups, setExpandedGroups] = useState<Partial<Record<CaseKind, boolean>>>({});
  const chartRef = useRef<ReactECharts>(null);
  const dateInputRef = useRef<HTMLInputElement>(null);

  async function refreshList() {
    const rows = await api<Manifest[]>("/api/datasets");
    setDatasets(rows);
    setSelectedId((current) => current ?? rows[0]?.dataset_id ?? null);
  }

  useEffect(() => {
    refreshList().catch((cause) => setError(cause.message));
  }, []);

  useEffect(() => {
    if (!selectedId) {
      setDetail(null);
      return;
    }
    setDetail(null);
    setCaseId(null);
    setCaseInput(null);
    setCaseTruth(null);
    setSelectedEventId(null);
    setTruthVisible(true);
    setView("observed");
    setExpandedGroups({});
    let disposed = false;
    let timer: number | undefined;
    const load = async () => {
      try {
        const next = await api<Manifest>(`/api/datasets/${selectedId}`);
        if (disposed) return;
        setDetail(next);
        setCaseId((current) =>
          next.cases.some((item) => item.case_id === current)
            ? current
            : (next.cases[0]?.case_id ?? null),
        );
        await refreshList();
        if (
          !disposed &&
          (next.status === "queued" || next.status === "running")
        ) {
          timer = window.setTimeout(load, 900);
        }
      } catch (cause) {
        if (!disposed) setError((cause as Error).message);
      }
    };
    load();
    return () => {
      disposed = true;
      window.clearTimeout(timer);
    };
  }, [selectedId]);

  useEffect(() => {
    if (
      !selectedId ||
      !caseId ||
      detail?.dataset_id !== selectedId ||
      !detail.cases.some((item) => item.case_id === caseId)
    ) {
      setCaseInput(null);
      setCaseTruth(null);
      setTruthVisible(true);
      return;
    }
    let disposed = false;
    setCaseInput(null);
    setCaseTruth(null);
    setSelectedEventId(null);
    setEventAxisFilter("all");
    setEventFamilyFilter("all");
    setTruthVisible(true);
    setView("observed");
    api<CaseInput>(`/api/datasets/${selectedId}/cases/${caseId}`)
      .then((input) => {
        if (!disposed) {
          setCaseInput(input);
          setDateTarget(input.dates[0]);
        }
      })
      .catch((cause) => {
        if (!disposed) setError(cause.message);
      });
    return () => {
      disposed = true;
    };
  }, [selectedId, caseId, detail?.dataset_id]);

  useEffect(() => {
    if (!truthVisible || !selectedId || !caseId || !caseInput) {
      setCaseTruth(null);
      setTruthLoading(false);
      return;
    }
    let disposed = false;
    setTruthLoading(true);
    api<CaseTruth>(`/api/datasets/${selectedId}/cases/${caseId}/truth`)
      .then((truth) => { if (!disposed) {
        setCaseTruth(truth);
        if (truth.events[0]) setComponentAxis(axisNames.indexOf(truth.events[0].axis) as 0 | 1 | 2);
      } })
      .catch((cause) => { if (!disposed) { setError(cause.message); setTruthVisible(false); } })
      .finally(() => { if (!disposed) setTruthLoading(false); });
    return () => { disposed = true; };
  }, [truthVisible, selectedId, caseId, caseInput]);

  const filteredCases = useMemo(
    () =>
      detail?.cases.filter((item) => item.case_id.includes(query.trim())) ?? [],
    [detail, query],
  );
  const groupedCases = useMemo(
    () => caseTypes.map((type) => ({
      type,
      cases: filteredCases.filter((item) => item.case_type === type),
    })).filter((group) => group.cases.length > 0),
    [filteredCases],
  );
  const selectedCaseType = detail?.dataset_id === selectedId
    ? detail.cases.find((item) => item.case_id === caseId)?.case_type
    : undefined;

  useEffect(() => {
    if (selectedCaseType) {
      setExpandedGroups((current) => ({ ...current, [selectedCaseType]: true }));
    }
  }, [selectedId, selectedCaseType]);

  const selectedEvent = caseTruth?.events.find((event) => event.event_id === selectedEventId);
  const filteredEvents = caseTruth?.events.filter((event) =>
    (eventAxisFilter === "all" || event.axis === eventAxisFilter) &&
    (eventFamilyFilter === "all" || event.type === eventFamilyFilter)) ?? [];

  const chartOption = useMemo<EChartsOption | null>(() => {
    if (!caseInput || (view === "components" && !caseTruth)) return null;
    const dates = caseInput.dates;
    const base = {
      animation: false,
      grid: { left: 58, right: 24, top: 34, bottom: 92 },
      tooltip: {
        trigger: "axis" as const,
        valueFormatter: (value: unknown) => valueLabel(Number(value)),
      },
      legend: {
        top: 0,
        right: 0,
        itemWidth: 12,
        itemHeight: 3,
        textStyle: { color: "#5c6966", fontSize: 12 },
      },
      xAxis: {
        type: "category" as const,
        data: dates,
        boundaryGap: false,
        axisLabel: { color: "#7d8986", hideOverlap: true },
        axisLine: { lineStyle: { color: "#ccd6d2" } },
      },
      yAxis: {
        type: "value" as const,
        name: "mm",
        nameTextStyle: { color: "#7d8986" },
        scale: true,
        axisLabel: { color: "#7d8986" },
        splitLine: { lineStyle: { color: "#e9eeeb" } },
      },
      dataZoom: [
        { type: "inside" as const, xAxisIndex: 0, filterMode: "none" as const },
        {
          type: "slider" as const,
          xAxisIndex: 0,
          bottom: 26,
          height: 18,
          brushSelect: false,
          fillerColor: "rgba(20,125,114,.12)",
          borderColor: "#d5e0db",
          handleStyle: { color: "#147d72" },
        },
      ],
    };
    if (view === "components") {
      const axis = axisNames[componentAxis];
      const truth = caseTruth!;
      const componentData: Record<ComponentKey, number[]> = {
        noise: truth.measurement_noise_mm.map((row) => row[componentAxis]),
        delta: truth.anomaly_delta_mm.map((row) => row[componentAxis]),
        observed: caseInput.observed_coordinate_mm.map((row) => row[componentAxis]),
      };
      return {
        ...base,
        series: [ ...(Object.keys(componentLabels) as ComponentKey[])
          .filter((key) => componentsVisible[key])
          .map((key) => ({
            name: `${axis} ${componentLabels[key]}`,
            type: "line" as const,
            showSymbol: false,
            lineStyle: { width: key === "observed" ? 2.2 : 1.6, color: componentColors[key] },
            itemStyle: { color: componentColors[key] },
            data: componentData[key],
          })) ],
      };
    }
    const series: LineSeriesOption[] = axisNames
      .filter((axis) => visible[axis])
      .map((axis) => {
        const index = axisNames.indexOf(axis);
        return {
          name: `${axis} 位移`,
          type: "line",
          showSymbol: false,
          smooth: false,
          lineStyle: { width: 2, color: colors[axis] },
          itemStyle: { color: colors[axis] },
          data: caseInput.displacement_mm.map((row) => row[index]),
        };
      });
    if (visible.H) {
      series.push({
        name: "H 水平偏移",
        type: "line",
        showSymbol: false,
        smooth: false,
        lineStyle: { width: 2, color: colors.H },
        itemStyle: { color: colors.H },
        data: caseInput.horizontal_offset_mm,
      });
    }
    if (showTruth && caseTruth) {
      axisNames
        .filter((axis) => visible[axis])
        .forEach((axis) => {
          const index = axisNames.indexOf(axis);
          series.push({
            name: `${axis} 配对正常观测`,
            type: "line",
            showSymbol: false,
            smooth: false,
            lineStyle: { width: 1.5, type: "dashed", color: colors[axis] },
            itemStyle: { color: colors[axis] },
            data: caseTruth.measurement_noise_mm.map((row) => row[index]),
          });
        });
    }
    if (caseTruth && series.length > 0) {
      const pointEvents = caseTruth.events.filter((event) => event.start_index === event.end_index);
      const intervalEvents = caseTruth.events.filter((event) => event.start_index !== event.end_index);
      series[0].markLine = {
        silent: true,
        symbol: "none",
        label: { show: false },
        data: pointEvents.map((event) => ({
          xAxis: event.start_date,
          lineStyle: {
            color: event.event_id === selectedEventId ? "#aa3e25" : "#b96b46",
            type: event.type === "global_extremum" ? "dashed" as const : "solid" as const,
            width: event.event_id === selectedEventId ? 2.5 : 1.4,
            opacity: selectedEventId && event.event_id !== selectedEventId ? 0.45 : 0.9,
          },
        })),
      };
      series[0].markArea = {
        silent: true,
        label: { show: false },
        data: intervalEvents.map((event) => [{
          xAxis: event.start_date,
          itemStyle: {
            color: event.event_id === selectedEventId
              ? "rgba(178, 76, 50, 0.23)"
              : selectedEventId ? "rgba(185, 107, 70, 0.06)" : "rgba(185, 107, 70, 0.12)",
          },
        }, { xAxis: event.end_date }]),
      };
    }
    return { ...base, series };
  }, [caseInput, caseTruth, view, visible, showTruth, componentAxis, componentsVisible, selectedEventId]);

  async function createDataset(event: React.FormEvent) {
    event.preventDefault();
    const parsedSeed = Number(seed);
    const parsedCount = Number(count);
    if (
      !Number.isInteger(parsedSeed) ||
      parsedSeed < 0 ||
      parsedSeed > 4294967295 ||
      !Number.isInteger(parsedCount) ||
      parsedCount < 1 ||
      parsedCount > 5000 ||
      (caseType === "all" && parsedCount % 4 !== 0)
    ) {
      setError(caseType === "all" ? "配对批次数量必须是 4 的倍数（推荐 24）；Seed 范围为 0～4294967295，案例数量最多 5000。" : "Seed 范围为 0～4294967295，案例数量为 1～5000。");
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      const created = await api<Manifest>("/api/datasets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seed: parsedSeed,
          count: parsedCount,
          case_type: caseType,
        }),
      });
      setSelectedId(created.dataset_id);
      setPage("datasets");
      setQuery("");
      await refreshList();
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  function locateDate() {
    if (!caseInput || !chartRef.current) return;
    const index = caseInput.dates.indexOf(
      dateInputRef.current?.value || dateTarget,
    );
    if (index < 0) {
      setError("所选日期不在当前案例中。");
      return;
    }
    setError("");
    const chart = chartRef.current.getEchartsInstance();
    const last = caseInput.dates.length - 1;
    chart.dispatchAction({
      type: "dataZoom",
      start: (Math.max(0, index - 12) * 100) / last,
      end: (Math.min(last, index + 12) * 100) / last,
    });
    chart.dispatchAction({ type: "hideTip" });
  }

  function resetZoom() {
    chartRef.current
      ?.getEchartsInstance()
      .dispatchAction({ type: "dataZoom", start: 0, end: 100 });
  }

  return (
    <div className={`app-shell ${page === "docs" ? "docs-mode" : ""}`}>
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">
            <Activity size={23} strokeWidth={2.2} />
          </span>
          <div>
            <strong>GNSS LAB</strong>
            <small>模拟实验台 / synthetic-v1</small>
          </div>
        </div>
        <nav className="main-nav" aria-label="主导航">
          <button
            className={page === "datasets" ? "active" : ""}
            onClick={() => setPage("datasets")}
          >
            <Database size={17} /> 数据集
          </button>
          <button
            className={page === "runs" ? "active" : ""}
            onClick={() => setPage("runs")}
          >
            <Layers3 size={17} /> 检测运行
          </button>
          <button
            className={page === "docs" ? "active" : ""}
            onClick={() => setPage("docs")}
          >
            <BookOpenText size={17} /> 实验文档
          </button>
        </nav>
        <div className="sidebar-section-title">
          <span>生成历史</span>
          <span>{datasets.length}</span>
        </div>
        <div className="dataset-list">
          {datasets.length === 0 && <p className="side-empty">尚无数据集</p>}
          {datasets.map((item) => (
            <button
              key={item.dataset_id}
              className={`dataset-item ${selectedId === item.dataset_id ? "selected" : ""}`}
              onClick={() => {
                setSelectedId(item.dataset_id);
                setPage("datasets");
                setQuery("");
              }}
            >
              <span className="dataset-item-top">
                <strong>{item.dataset_id}</strong>
                <ChevronRight size={14} />
              </span>
              <span className="dataset-item-meta">
                {item.generated_cases}/{item.request.count} 例 <i /> Seed{" "}
                {item.request.seed} · {generationTypeLabels[item.request.case_type]}
              </span>
              <span className={`status-pill ${item.status}`}>
                {statusLabels[item.status]}
              </span>
            </button>
          ))}
        </div>
        <div className="sidebar-foot">
          <span className="foot-indicator" /> 本地合成数据
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumb">
            实验 /{" "}
            <strong>{page === "datasets" ? "数据集" : page === "runs" ? "检测运行" : "实验文档"}</strong>
          </div>
          <div className="topbar-meta">
            <span>日尺度</span>
            <span>N / E / U</span>
            <span>单位 mm</span>
          </div>
        </header>
        {error && (
          <div className="error-banner" role="alert">
            {error}
            <button onClick={() => setError("")} aria-label="关闭错误">
              ×
            </button>
          </div>
        )}

        {page === "docs" ? (
          <DocsPage />
        ) : page === "runs" ? (
          <section className="runs-page">
            <div className="page-heading">
              <div>
                <p className="eyebrow">DETECTION RUNS</p>
                <h1>检测运行</h1>
              </div>
            </div>
            <div className="runs-empty">
              <p className="eyebrow">EXPERIMENT WORKFLOW</p>
              <h2>暂无有效检测结果</h2>
              <p>全部旧实验结果已作废。新实验需要独立 Normal 校准、明确的数据划分和运行登记。</p>
              <p>数值与视觉独立处理相同观测；运行完成后，通过本地 comparison.html 核对指标、真值和逐例输出。</p>
              <div className="runs-links">
                <a href="?view=docs&doc=detection"><strong>检测与评价规则 <ChevronRight size={17} /></strong><span>输入隔离 · Point / Range · 请求预算与失败统计</span></a>
                <a href="?view=docs&doc=development"><strong>运行与开发说明 <ChevronRight size={17} /></strong><span>校准 · 登记 · 检测 · 离线报告</span></a>
              </div>
              <small>网页不发起模型调用。结果由命令行生成，不预置历史成绩。</small>
            </div>
          </section>
        ) : (
          <>
            <section className="page-heading">
              <div>
                <p className="eyebrow">SOURCE-VERIFIED / SYNTHETIC GNSS</p>
                <h1>原生生成 · 配对检查</h1>
                <p className="subtitle">GutenTAG × TODS · 独立高斯背景 · 365 日 N/E/U</p>
              </div>
              <div className="heading-badge">
                <span className="badge-dot" /> Point / Range
              </div>
            </section>

            <section className="create-band" aria-labelledby="create-title">
              <div className="create-intro">
                <span className="section-icon">
                  <Play size={18} fill="currentColor" />
                </span>
                <div>
                  <h2 id="create-title">生成数据集</h2>
                  <p>Normal + 三类异常 · 同背景逐值配对</p>
                </div>
              </div>
              <form onSubmit={createDataset} className="create-form">
                <label>
                  案例类型
                  <select value={caseType} onChange={(event) => setCaseType(event.target.value as GenerationType)}>
                    {(["all", ...caseTypes] as GenerationType[]).map((type) => (
                      <option key={type} value={type}>{generationTypeLabels[type]}</option>
                    ))}
                  </select>
                </label>
                <label>
                  Seed
                  <input
                    type="number"
                    min="0"
                    max="4294967295"
                    step="1"
                    value={seed}
                    onChange={(event) => setSeed(event.target.value)}
                  />
                </label>
                <label>
                  案例数
                  <input
                    type="number"
                    min={caseType === "all" ? "4" : "1"}
                    max="5000"
                    step="1"
                    value={count}
                    onChange={(event) => setCount(event.target.value)}
                  />
                </label>
                <button
                  className="primary-button"
                  type="submit"
                  disabled={submitting}
                >
                  {submitting ? (
                    <LoaderCircle className="spin" size={17} />
                  ) : (
                    <Play size={17} fill="currentColor" />
                  )}
                  <span>{submitting ? "提交中" : "开始生成"}</span>
                </button>
              </form>
              {caseType === "all" && <p className="mix-note">每 4 例共用一份背景 · 24 例覆盖 N/E/U × 正负方向 · 工程检查样例</p>}
            </section>

            <section className="fixed-protocol" aria-label="固定生成参数">
              <div className="fixed-protocol-heading">
                <span>CONTROLLED PROTOCOL / UPSTREAM-CORE-V1</span>
                <strong>固定生成参数</strong>
              </div>
              <dl>
                <div><dt>长度</dt><dd>365 日</dd></div>
                <div><dt>起始日期</dt><dd>2025-01-01</dd></div>
                <div><dt>正常均值 · N/E/U</dt><dd>0 / 0 / 0 mm</dd></div>
                <div><dt>Range 偏移 · N/E/U</dt><dd>3σ = 1.5 / 1.5 / 3.0 mm</dd></div>
                <div><dt>噪声 σ · N/E/U</dt><dd>0.5 / 0.5 / 1.0 mm</dd></div>
              </dl>
              <p>极值强度由 GutenTAG 按噪声自动生成；Range 参数为本项目选择。保留自然高斯尾部，不按曲线重抽。</p>
            </section>

            <section className="morphology-guide" aria-label="异常规则与标签">
              <div><small>REFERENCE / GUTENTAG</small><h3>Normal</h3><code>Polynomial [0] + noise</code><p>零基底与原生高斯噪声。组内异常案例逐值复用这份背景。</p></div>
              <div><small>POINT / GUTENTAG</small><h3>Global Extremum</h3><code>length = 1 · local = false</code><p>原类自动保护极值幅度，再叠加当天噪声。标签为单日。</p></div>
              <div><small>RANGE / TODS</small><h3>Trend</h3><code>radius = 45 · |factor| = 3/89</code><p>90 日线性累计后保留位移；仅适配位置接口，标签来自原函数。</p></div>
              <div><small>RANGE / GUTENTAG</small><h3>Transient Mean Shift</h3><code>length = 14 · offset = ±3</code><p>原生 mean 保留噪声，区间后恢复；逐轴标签来自原生协议。</p></div>
            </section>
            <p className="protocol-reference">上游实现已锁定 commit、许可证与文件哈希；参数和毫米映射属于项目配置。<a href="?view=docs&doc=data-generation">查看复用边界与完整协议 →</a></p>

            {!detail ? (
              <div className="welcome-empty">
                <Database size={32} strokeWidth={1.3} />
                <h2>尚无数据集</h2>
              </div>
            ) : (
              <>
                <section className="dataset-heading">
                  <div>
                    <p className="eyebrow">DATASET / {detail.dataset_id}</p>
                    <h2>生成批次</h2>
                    <p>
                      {formattedDate(detail.created_at)} · Seed{" "}
                      {detail.request.seed} · {detail.generator_version}
                      {" · "}{generationTypeLabels[detail.request.case_type]}
                    </p>
                  </div>
                  <span className={`large-status ${detail.status}`}>
                    {detail.status === "running" && (
                      <LoaderCircle className="spin" size={16} />
                    )}
                    {statusLabels[detail.status]}
                  </span>
                </section>
                <div className="type-counts" aria-label="计划类型数量">
                  {caseTypes.filter((type) => detail.type_counts[type]).map((type) => (
                    <span key={type}>{kindLabels[type]} <strong>{detail.type_counts[type]}</strong></span>
                  ))}
                </div>
                <div className="stats-row">
                  <div>
                    <small>案例进度</small>
                    <strong>
                      {detail.generated_cases}
                      <em> / {detail.request.count}</em>
                    </strong>
                  </div>
                  <div>
                    <small>观测跨度</small>
                    <strong>
                      365 <em>天</em>
                    </strong>
                  </div>
                  <div>
                    <small>参考坐标</small>
                    <strong className="reference-stat">
                      0 / 0 / 0{" "}
                      <em>mm</em>
                    </strong>
                  </div>
                  <div>
                    <small>注入事件</small>
                    <strong>{detail.cases.reduce((sum, item) => sum + item.event_count, 0)}</strong>
                  </div>
                </div>
                {(detail.status === "queued" ||
                  detail.status === "running") && (
                  <div
                    className="progress-track"
                    role="progressbar"
                    aria-valuenow={detail.generated_cases}
                    aria-valuemin={0}
                    aria-valuemax={detail.request.count}
                  >
                    <span
                      style={{
                        width: `${(detail.generated_cases / detail.request.count) * 100}%`,
                      }}
                    />
                  </div>
                )}
                {detail.status === "failed" && (
                  <div className="error-banner">
                    {detail.error || "生成失败"}
                  </div>
                )}

                <div className="workspace-grid">
                  <section className="case-browser">
                    <div className="panel-title">
                      <div>
                        <p className="eyebrow">CASE INDEX</p>
                        <h2>案例列表</h2>
                      </div>
                      <span className="count-label">
                        {filteredCases.length}
                      </span>
                    </div>
                    <label className="search-field">
                      <Search size={16} />
                      <input
                        value={query}
                        onChange={(event) => setQuery(event.target.value)}
                        placeholder="搜索案例 ID"
                        aria-label="搜索案例 ID"
                      />
                    </label>
                    <div className="case-list">
                      {groupedCases.map(({ type, cases }) => {
                        const expanded = Boolean(query.trim()) || Boolean(expandedGroups[type]);
                        return <div className="case-group" key={type}>
                          <button className="case-group-toggle" aria-expanded={expanded}
                            onClick={() => setExpandedGroups((current) => ({ ...current, [type]: !expanded }))}>
                            <span>{kindLabels[type]} <small>{cases.length}</small></span>
                            <ChevronDown size={16} className={expanded ? "expanded" : ""} />
                          </button>
                          {expanded && cases.map((item) => (
                            <button
                              className={`case-row ${caseId === item.case_id ? "selected" : ""}`}
                              key={item.case_id}
                              onClick={() => setCaseId(item.case_id)}
                            >
                              <span className="case-index">{item.case_id.slice(-4)}</span>
                              <span>
                                <strong>{item.case_id}</strong>
                                <small>{item.group_axis}{item.group_sign > 0 ? "+" : "−"} · {item.background_group.replace("group_", "组 ")}</small>
                              </span>
                              <ChevronRight size={16} />
                            </button>
                          ))}
                        </div>;
                      })}
                      {filteredCases.length === 0 && (
                        <p className="list-empty">没有匹配的案例</p>
                      )}
                    </div>
                  </section>

                  <section className="case-detail">
                    <div className="panel-title detail-title">
                      <div>
                        <p className="eyebrow">OBSERVATION SERIES</p>
                        <h2>{caseId || "等待案例"}</h2>
                      </div>
                      <div className="case-tag">
                        <Check size={14} /> {selectedCaseType ? kindLabels[selectedCaseType] : "等待案例"}
                      </div>
                    </div>
                    {caseTruth && detail.request.case_type === "all" && <div className="paired-switcher">
                      <span>同背景 · {caseTruth.background_group.replace("group_", "组 ")}</span>
                      <div>{detail.cases.filter((item) => item.background_group === caseTruth.background_group).map((item) =>
                        <button key={item.case_id} className={item.case_id === caseId ? "active" : ""} onClick={() => setCaseId(item.case_id)}>{kindLabels[item.case_type]}</button>)}</div>
                    </div>}
                    {caseInput && <div className="truth-access">
                      <span>{truthVisible ? caseTruth ? "事件标注已显示" : "正在读取事件标注…" : "事件标注已隐藏"}</span>
                      <button className="truth-command" onClick={() => {
                        if (truthVisible) { setView("observed"); setShowTruth(false); }
                        setTruthVisible((old) => !old);
                      }}
                        disabled={truthLoading} aria-pressed={truthVisible}>
                        {truthLoading ? "读取中…" : truthVisible ? "隐藏标注" : "显示标注"}
                      </button>
                    </div>}
                    {!caseInput || !chartOption ? (
                      <div className="chart-loading">
                        {!caseInput ? (caseId ? "正在读取曲线…" : "等待案例生成…") : "正在读取生成成分…"}
                      </div>
                    ) : (
                      <>
                        <div className="chart-toolbar">
                          <div
                            className="segmented"
                            role="group"
                            aria-label="图表视图"
                          >
                            <button
                              className={view === "observed" ? "active" : ""}
                              onClick={() => setView("observed")}
                            >
                              观测位移
                            </button>
                            <button
                              className={view === "components" ? "active" : ""}
                              onClick={() => { setTruthVisible(true); setView("components"); }}
                            >
                              生成成分
                            </button>
                          </div>
                          {view === "observed" ? (
                            <div className="axis-toggles">
                              {(["N", "E", "U", "H"] as const).map((axis) => (
                                <label key={axis}>
                                  <input
                                    type="checkbox"
                                    checked={visible[axis]}
                                    onChange={() =>
                                      setVisible((old) => ({
                                        ...old,
                                        [axis]: !old[axis],
                                      }))
                                    }
                                  />
                                  <span
                                    className={`axis-swatch axis-${axis}`}
                                  />
                                  {axis}
                                </label>
                              ))}
                            </div>
                          ) : (
                            <div className="component-controls">
                              <div className="segmented axis-segment" role="group" aria-label="成分分量">
                                {axisNames.map((axis, index) => (
                                  <button className={componentAxis === index ? "active" : ""}
                                    key={axis} onClick={() => setComponentAxis(index as 0 | 1 | 2)}>{axis}</button>
                                ))}
                              </div>
                              <div className="component-toggles" role="group" aria-label="显示成分">
                                {(Object.keys(componentLabels) as ComponentKey[]).map((key) => (
                                  <label key={key}>
                                    <input type="checkbox" checked={componentsVisible[key]}
                                      onChange={() => setComponentsVisible((old) => ({ ...old, [key]: !old[key] }))} />
                                    <span style={{ background: componentColors[key] }} />{componentLabels[key]}
                                  </label>
                                ))}
                              </div>
                            </div>
                          )}
                        </div>
                        <div className="chart-surface">
                          <ReactECharts
                            ref={chartRef}
                            option={chartOption}
                            style={{ height: 390, width: "100%" }}
                            notMerge
                          />
                        </div>
                        <div className="chart-actions">
                          <label className="date-control">
                            <CalendarDays size={16} />
                            <input
                              ref={dateInputRef}
                              type="date"
                              min={caseInput.dates[0]}
                              max={caseInput.dates.at(-1)}
                              value={dateTarget}
                              onChange={(event) =>
                                setDateTarget(event.target.value)
                              }
                              aria-label="定位日期"
                            />
                          </label>
                          <button
                            className="icon-command"
                            onClick={locateDate}
                            title="定位日期"
                            aria-label="定位日期"
                          >
                            <Crosshair size={17} />
                          </button>
                          <button
                            className="icon-command"
                            onClick={resetZoom}
                            title="重置缩放"
                            aria-label="重置缩放"
                          >
                            <RotateCcw size={17} />
                          </button>
                          {view === "observed" && caseTruth && (
                            <label className="truth-toggle">
                              <input
                                type="checkbox"
                                checked={showTruth}
                                onChange={() => setShowTruth((old) => !old)}
                              />
                              配对正常观测
                            </label>
                          )}
                        </div>
                        <div className="case-footer">
                          <div>
                            <small>日期</small>
                            <strong>
                              {caseInput.dates[0]} — {caseInput.dates.at(-1)}
                            </strong>
                          </div>
                          <div>
                            <small>最大水平偏移 H</small>
                            <strong>
                              {valueLabel(
                                Math.max(...caseInput.horizontal_offset_mm),
                              )}
                            </strong>
                          </div>
                          <div>
                            <small>最大三维偏移 R3D</small>
                            <strong>
                              {valueLabel(
                                Math.max(...caseInput.spatial_offset_mm),
                              )}
                            </strong>
                          </div>
                        </div>
                        <div className="events-line">
                          <span>
                            <Filter size={15} /> 事件标签时间线
                          </span>
                          <strong>{!caseTruth ? truthVisible ? "读取中" : "标注已隐藏" : caseTruth.events.length === 0 ? "无注入事件" : `${caseTruth.events.length} 个事件`}</strong>
                        </div>
                        {caseTruth && caseTruth.events.length > 0 && <>
                          <div className="timeline-filters" aria-label="事件筛选">
                            <div className="segmented" role="group" aria-label="按轴筛选">
                              {(["all", ...axisNames] as const).map((axis) => <button key={axis} className={eventAxisFilter === axis ? "active" : ""} onClick={() => { setEventAxisFilter(axis); setSelectedEventId(null); }}>{axis === "all" ? "全部轴" : axis}</button>)}
                            </div>
                            <div className="segmented" role="group" aria-label="按事件族筛选">
                              {([ ["all", "全部"], ["global_extremum", "极值"], ["trend", "趋势"], ["mean_shift", "暂态均值"] ] as const).map(([value, label]) => <button key={value} className={eventFamilyFilter === value ? "active" : ""} onClick={() => { setEventFamilyFilter(value); setSelectedEventId(null); }}>{label}</button>)}
                            </div>
                          </div>
                          <div className="event-timeline">
                            {filteredEvents.map((event) => <button key={event.event_id} className={`timeline-row ${selectedEventId === event.event_id ? "active" : ""} ${selectedEventId && selectedEventId !== event.event_id ? "dimmed" : ""}`} onClick={() => {
                              setSelectedEventId(selectedEventId === event.event_id ? null : event.event_id);
                              setComponentAxis(axisNames.indexOf(event.axis) as 0 | 1 | 2);
                            }} aria-pressed={selectedEventId === event.event_id}>
                              <span className="timeline-label"><b>{event.event_id}</b> {caseTypeLabels[event.type]} · {event.axis}</span>
                              <span className="event-track"><i className={event.start_index === event.end_index ? "event-dot" : "event-band"} style={{ left: `${event.start_index / 364 * 100}%`, width: event.start_index === event.end_index ? undefined : `${(event.end_index - event.start_index + 1) / 365 * 100}%` }} /></span>
                              <span className="timeline-days">{event.start_index + 1}{event.end_index > event.start_index ? `–${event.end_index + 1}` : ""}</span>
                            </button>)}
                            {filteredEvents.length === 0 && <p className="list-empty">当前筛选下没有事件</p>}
                          </div>
                          {selectedEvent && <div className="event-inspector">
                            <div className="inspector-heading"><strong>{selectedEvent.event_id} · {caseTypeLabels[selectedEvent.type]} · {selectedEvent.axis}</strong><button onClick={() => setView("components")}>查看净改变量</button></div>
                            <dl>
                              <div><dt>开始</dt><dd>Day {selectedEvent.start_index + 1} · {selectedEvent.start_date}</dd></div>
                              <div><dt>结束</dt><dd>Day {selectedEvent.end_index + 1} · {selectedEvent.end_date}</dd></div>
                              <div><dt>持续 / 保留偏移</dt><dd>{selectedEvent.end_index - selectedEvent.start_index + 1} 日 / {selectedEvent.persistent ? "是" : "否"}</dd></div>
                              <div><dt>生成操作</dt><dd>{selectedEvent.operation === "native_extremum" ? "原生极值 + 当日噪声" : "叠加偏移"}</dd></div>
                              <div><dt>末端净改变量</dt><dd>{valueLabel(selectedEvent.target_offset_mm)} · {(selectedEvent.target_offset_mm / selectedEvent.sigma_mm).toFixed(2)}σ</dd></div>
                              <div><dt>末端实际观测</dt><dd>{valueLabel(selectedEvent.observed_end_mm)} · {(selectedEvent.observed_end_mm / selectedEvent.sigma_mm).toFixed(2)}σ</dd></div>
                              <div><dt>该轴噪声 σ</dt><dd>{valueLabel(selectedEvent.sigma_mm)}</dd></div>
                            </dl>
                            <p className="native-source">{selectedEvent.source}</p>
                            <code className="native-parameters">{JSON.stringify(selectedEvent.native_parameters)}</code>
                          </div>}
                        </>}
                        {caseTruth && <details className="seed-details">
                          <summary>生成 seed（仅用于核对真值）</summary>
                          <div><span>独立高斯噪声 seed</span><code>{caseTruth.noise_seed}</code></div>
                          <div><span>位置 / TODS seed</span><code>{caseTruth.position_seed} / {caseTruth.tods_seed}</code></div>
                          <p>净改变量 = 最终观测 − 配对背景。极值的净改变量和实际观测都不固定为 ±3σ。</p>
                        </details>}
                      </>
                    )}
                  </section>
                </div>
              </>
            )}
          </>
        )}
      </main>
    </div>
  );
}
