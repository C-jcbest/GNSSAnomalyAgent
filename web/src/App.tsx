import { useEffect, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import type { LineSeriesOption } from "echarts/charts";
import {
  Activity,
  CalendarDays,
  Check,
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

type Triple = [number, number, number];
type Status = "queued" | "running" | "complete" | "failed";
type Manifest = {
  dataset_id: string;
  created_at: string;
  status: Status;
  request: {
    preset: string;
    seed: number;
    count: number;
    config: { start_date: string; reference_coordinate_mm: Triple };
  };
  config_sha256: string;
  generated_cases: number;
  cases: {
    case_id: string;
    case_seed: number;
    days: number;
    event_count: number;
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
  events: unknown[];
  background_displacement_mm: Triple[];
  white_noise_mm: Triple[];
  ar_noise_mm: Triple[];
  observation_noise_mm: Triple[];
  true_coordinate_mm: Triple[];
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
    throw new Error(body.detail || `请求失败 (${response.status})`);
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
  const [seed, setSeed] = useState("20260923");
  const [count, setCount] = useState("10");
  const [submitting, setSubmitting] = useState(false);
  const [error, setError] = useState("");
  const [page, setPage] = useState<"datasets" | "runs">("datasets");
  const [view, setView] = useState<"observed" | "components">("observed");
  const [visible, setVisible] = useState({
    N: true,
    E: true,
    U: true,
    H: false,
  });
  const [componentAxis, setComponentAxis] = useState<0 | 1 | 2>(0);
  const [showTruth, setShowTruth] = useState(false);
  const [dateTarget, setDateTarget] = useState("");
  const [query, setQuery] = useState("");
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
      return;
    }
    let disposed = false;
    setCaseInput(null);
    setCaseTruth(null);
    Promise.all([
      api<CaseInput>(`/api/datasets/${selectedId}/cases/${caseId}`),
      api<CaseTruth>(`/api/datasets/${selectedId}/cases/${caseId}/truth`),
    ])
      .then(([input, truth]) => {
        if (!disposed) {
          setCaseInput(input);
          setCaseTruth(truth);
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

  const filteredCases = useMemo(
    () =>
      detail?.cases.filter((item) => item.case_id.includes(query.trim())) ?? [],
    [detail, query],
  );

  const chartOption = useMemo<EChartsOption | null>(() => {
    if (!caseInput || !caseTruth) return null;
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
      return {
        ...base,
        series: [
          {
            name: `${axis} 背景`,
            type: "line",
            showSymbol: false,
            lineStyle: { width: 2, color: colors.N },
            itemStyle: { color: colors.N },
            data: caseTruth.background_displacement_mm.map(
              (row) => row[componentAxis],
            ),
          },
          {
            name: `${axis} 白噪声`,
            type: "line",
            showSymbol: false,
            lineStyle: { width: 1.4, color: colors.E },
            itemStyle: { color: colors.E },
            data: caseTruth.white_noise_mm.map((row) => row[componentAxis]),
          },
          {
            name: `${axis} AR(1)`,
            type: "line",
            showSymbol: false,
            lineStyle: { width: 1.4, color: colors.U },
            itemStyle: { color: colors.U },
            data: caseTruth.ar_noise_mm.map((row) => row[componentAxis]),
          },
        ],
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
    if (showTruth) {
      axisNames
        .filter((axis) => visible[axis])
        .forEach((axis) => {
          const index = axisNames.indexOf(axis);
          series.push({
            name: `${axis} 无噪声`,
            type: "line",
            showSymbol: false,
            smooth: false,
            lineStyle: { width: 1.5, type: "dashed", color: colors[axis] },
            itemStyle: { color: colors[axis] },
            data: caseTruth.true_coordinate_mm.map(
              (row) => row[index] - caseInput.reference_coordinate_mm[index],
            ),
          });
        });
    }
    return { ...base, series };
  }, [caseInput, caseTruth, view, visible, showTruth, componentAxis]);

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
      parsedCount > 75
    ) {
      setError("Seed 范围为 0～4294967295，案例数量为 1～75。");
      return;
    }
    setSubmitting(true);
    setError("");
    try {
      const created = await api<Manifest>("/api/datasets", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          preset: "normal-p1",
          seed: parsedSeed,
          count: parsedCount,
        }),
      });
      setSelectedId(created.dataset_id);
      setPage("datasets");
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
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">
            <Activity size={23} strokeWidth={2.2} />
          </span>
          <div>
            <strong>GNSS LAB</strong>
            <small>模拟实验台 / P1</small>
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
                {item.request.seed}
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
            <strong>{page === "datasets" ? "数据集" : "检测运行"}</strong>
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

        {page === "runs" ? (
          <section className="runs-page">
            <div className="page-heading">
              <div>
                <p className="eyebrow">DETECTION RUNS</p>
                <h1>检测运行</h1>
              </div>
            </div>
            <div className="empty-runs">
              <Layers3 size={34} strokeWidth={1.3} />
              <h2>尚无检测运行</h2>
            </div>
          </section>
        ) : (
          <>
            <section className="page-heading">
              <div>
                <p className="eyebrow">SYNTHETIC DATA / NORMAL-P1</p>
                <h1>日坐标模拟数据</h1>
                <p className="subtitle">固定参考坐标 · 180 日窗口</p>
              </div>
              <div className="heading-badge">
                <span className="badge-dot" /> P1 正常序列
              </div>
            </section>

            <section className="create-band" aria-labelledby="create-title">
              <div className="create-intro">
                <span className="section-icon">
                  <Play size={18} fill="currentColor" />
                </span>
                <div>
                  <h2 id="create-title">生成数据集</h2>
                  <p>正常序列 / 白噪声 + AR(1) + 周期背景</p>
                </div>
              </div>
              <form onSubmit={createDataset} className="create-form">
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
                    min="1"
                    max="75"
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
            </section>

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
                      {detail.request.seed} · 配置{" "}
                      {detail.config_sha256.slice(0, 12)}
                    </p>
                  </div>
                  <span className={`large-status ${detail.status}`}>
                    {detail.status === "running" && (
                      <LoaderCircle className="spin" size={16} />
                    )}
                    {statusLabels[detail.status]}
                  </span>
                </section>
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
                      180 <em>天</em>
                    </strong>
                  </div>
                  <div>
                    <small>参考坐标</small>
                    <strong className="reference-stat">
                      {detail.request.config.reference_coordinate_mm.join(
                        " / ",
                      )}{" "}
                      <em>mm</em>
                    </strong>
                  </div>
                  <div>
                    <small>注入事件</small>
                    <strong>0</strong>
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
                      {filteredCases.map((item, index) => (
                        <button
                          className={`case-row ${caseId === item.case_id ? "selected" : ""}`}
                          key={item.case_id}
                          onClick={() => setCaseId(item.case_id)}
                        >
                          <span className="case-index">
                            {String(index + 1).padStart(2, "0")}
                          </span>
                          <span>
                            <strong>{item.case_id}</strong>
                            <small>180 天 · 0 个事件</small>
                          </span>
                          <ChevronRight size={16} />
                        </button>
                      ))}
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
                        <Check size={14} /> 正常序列
                      </div>
                    </div>
                    {!caseInput || !caseTruth || !chartOption ? (
                      <div className="chart-loading">
                        {caseId ? "正在读取曲线…" : "等待案例生成…"}
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
                              onClick={() => setView("components")}
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
                            <div
                              className="segmented axis-segment"
                              role="group"
                              aria-label="成分分量"
                            >
                              {axisNames.map((axis, index) => (
                                <button
                                  className={
                                    componentAxis === index ? "active" : ""
                                  }
                                  key={axis}
                                  onClick={() =>
                                    setComponentAxis(index as 0 | 1 | 2)
                                  }
                                >
                                  {axis}
                                </button>
                              ))}
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
                          {view === "observed" && (
                            <label className="truth-toggle">
                              <input
                                type="checkbox"
                                checked={showTruth}
                                onChange={() => setShowTruth((old) => !old)}
                              />
                              无噪声轨迹
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
                            <small>最大三维偏移 D</small>
                            <strong>
                              {valueLabel(
                                Math.max(...caseInput.spatial_offset_mm),
                              )}
                            </strong>
                          </div>
                        </div>
                        <div className="events-line">
                          <span>
                            <Filter size={15} /> 异常标注
                          </span>
                          <strong>
                            {caseTruth.events.length === 0
                              ? "无注入事件"
                              : `${caseTruth.events.length} 个事件`}
                          </strong>
                        </div>
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
