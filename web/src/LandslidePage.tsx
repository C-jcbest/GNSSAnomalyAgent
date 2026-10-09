import { useEffect, useId, useMemo, useRef, useState } from "react";
import ReactECharts from "echarts-for-react";
import type { EChartsOption } from "echarts";
import { connect, disconnect } from "echarts";
import type { LineSeriesOption } from "echarts/charts";
import {
  ArrowDownToLine,
  ArrowUpRight,
  ChevronLeft,
  ChevronRight,
  LoaderCircle,
  Mountain,
  Plus,
  RotateCcw,
} from "lucide-react";
import "./landslide.css";

type Triple = [number, number, number];
type Scenario =
  | "mixed_stages"
  | "stable"
  | "slow_creep"
  | "seasonal_steps"
  | "progressive_acceleration"
  | "acceleration_arrest"
  | "reactivation";
type Manifest = {
  dataset_id: string;
  created_at: string;
  status: "queued" | "running" | "complete" | "failed";
  request: { seed: number; count: number; days: number; start_date: string };
  generated_cases: number;
  type_counts: Record<string, number>;
  observed_extent_mm: [number, number];
  cases: {
    case_id: string;
    scenario: Scenario;
    observed_days: number;
    missing_days: number;
  }[];
  error: string | null;
};
type Observation = {
  case_id: string;
  dates: string[];
  displacement_mm: (Triple | null)[];
};
type Truth = {
  case_id: string;
  latent_displacement_mm: Triple[];
  phases: {
    kind: string;
    start_index: number;
    end_index: number;
    description: string;
  }[];
};
type Diagnostics = {
  case_id: string;
  source: "observations_only";
  window_days: 31 | 61 | 91;
  velocity_mm_day: (Triple | null)[];
  acceleration_mm_day2: (Triple | null)[];
  speed_mm_day: (number | null)[];
  tangential_acceleration_mm_day2: (number | null)[];
};
const scenarioLabels: Record<Scenario, string> = {
  mixed_stages: "复合阶段",
  stable: "稳定对照",
  slow_creep: "缓慢蠕变",
  seasonal_steps: "季节阶梯",
  progressive_acceleration: "持续加速",
  acceleration_arrest: "加速后回稳",
  reactivation: "停滞后再活动",
};
const phaseLabels: Record<string, string> = {
  stable: "稳定",
  creep: "蠕变",
  acceleration: "加速",
  steady_slip: "持续滑移",
  deceleration: "减速",
  dormant: "停滞",
  seasonal_activity: "季节活动",
  reactivation: "再活动",
};
const phaseColors: Record<string, string> = {
  stable: "#b0bca6",
  creep: "#76a58c",
  acceleration: "#cf805e",
  steady_slip: "#b8a15e",
  deceleration: "#7b9ba7",
  dormant: "#d2d8c8",
  seasonal_activity: "#a391b0",
  reactivation: "#b88b70",
};
const statusLabels = {
  queued: "排队中",
  running: "生成中",
  complete: "已完成",
  failed: "生成失败",
};
const axes = ["N", "E", "U"];
const axisColors = ["#147d72", "#cc6d4b", "#a38a32"];
const basePath = "/api/landslide-datasets";

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const response = await fetch(url, { cache: "no-store", ...options });
  if (!response.ok) {
    const body = await response.json().catch(() => ({}));
    if (typeof body.detail === "string") {
      throw new Error(body.detail);
    }
    if (Array.isArray(body.detail)) {
      throw new Error(
        body.detail.map((item: { msg: string }) => item.msg).join("；"),
      );
    }
    throw new Error(`请求失败 (${response.status})`);
  }
  return response.json();
}

function observedExtent(observation: Observation): [number, number] {
  let lower = 0;
  let upper = 0;
  for (const point of observation.displacement_mm) {
    if (point === null) {
      continue;
    }
    for (const value of point) {
      lower = Math.min(lower, value);
      upper = Math.max(upper, value);
    }
  }
  return [lower, upper];
}

function paddedExtent(
  extent: [number, number],
  minimumSpan: number,
  paddingPercent: number,
): [number, number] {
  const span = Math.max(extent[1] - extent[0], minimumSpan);
  const middle = (extent[0] + extent[1]) / 2;
  const halfRange = span * (0.5 + paddingPercent / 100);
  return [middle - halfRange, middle + halfRange];
}

// 保留缺测日的空值；居中窗口只用于离线阅读，不表示可实时获得的预测信号。
function rollingMedian(points: (Triple | null)[]): (Triple | null)[] {
  return points.map((point, index) => {
    if (point === null) {
      return null;
    }
    const neighborhood = points.slice(Math.max(0, index - 15), index + 16);
    return [0, 1, 2].map((axis) => {
      const values = neighborhood
        .filter((sample): sample is Triple => sample !== null)
        .map((sample) => sample[axis])
        .sort((a, b) => a - b);
      const middle = Math.floor(values.length / 2);
      if (values.length % 2 === 0) {
        return (values[middle - 1] + values[middle]) / 2;
      }
      return values[middle];
    }) as Triple;
  });
}

function curvePath(
  points: (Triple | null)[],
  axis: number,
  extent: [number, number],
) {
  let connected = false;
  return points
    .map((point, index) => {
      if (point === null) {
        connected = false;
        return "";
      }
      const x = 12 + (index / Math.max(1, points.length - 1)) * 336;
      const y =
        10 + ((extent[1] - point[axis]) / (extent[1] - extent[0])) * 102;
      const command = connected ? "L" : "M";
      connected = true;
      return `${command}${x.toFixed(1)},${y.toFixed(1)}`;
    })
    .join("");
}

function OverviewCurve({
  observation,
  extent,
}: {
  observation: Observation;
  extent: [number, number];
}) {
  const median = useMemo(
    () => rollingMedian(observation.displacement_mm),
    [observation],
  );
  const zero = 10 + (extent[1] / (extent[1] - extent[0])) * 102;
  return (
    <svg
      className="ls-mini-chart"
      viewBox="0 0 360 126"
      role="img"
      aria-label={`${observation.case_id} 的 N、E、U 位移；淡线原始观测，实线31日中位数`}
    >
      {[30, 60, 90].map((y) => (
        <line key={y} x1="12" x2="348" y1={y} y2={y} stroke="#e9ede7" />
      ))}
      <line
        x1="12"
        x2="348"
        y1={zero}
        y2={zero}
        stroke="#c5cfc7"
        strokeDasharray="3 3"
      />
      {axes.map((axis, index) => (
        <g key={axis}>
          <path
            d={curvePath(observation.displacement_mm, index, extent)}
            fill="none"
            stroke={axisColors[index]}
            opacity="0.2"
            strokeWidth="0.7"
          />
          <path
            d={curvePath(median, index, extent)}
            fill="none"
            stroke={axisColors[index]}
            strokeWidth="1.7"
          />
        </g>
      ))}
    </svg>
  );
}

function DiagnosticsPanel({
  observation,
  datasetId,
  visibleAxes,
  chartGroup,
  observationChart,
}: {
  observation: Observation;
  datasetId: string;
  visibleAxes: boolean[];
  chartGroup: string;
  observationChart: React.RefObject<ReactECharts>;
}) {
  const [windowDays, setWindowDays] = useState<31 | 61 | 91>(61);
  const [diagnostics, setDiagnostics] = useState<Diagnostics | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [retry, setRetry] = useState(0);
  const endpoint = `${basePath}/${datasetId}/cases/${observation.case_id}`;

  useEffect(() => {
    const controller = new AbortController();
    setDiagnostics(null);
    setLoading(true);
    setError("");
    request<Diagnostics>(`${endpoint}/diagnostics?window_days=${windowDays}`, {
      signal: controller.signal,
    })
      .then((result) => {
        if (!controller.signal.aborted) {
          setDiagnostics(result);
        }
      })
      .catch((cause: Error) => {
        if (!controller.signal.aborted) {
          setError(cause.message);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setLoading(false);
        }
      });
    return () => controller.abort();
  }, [endpoint, windowDays, retry]);

  const chartOption = useMemo<EChartsOption | null>(() => {
    if (!diagnostics || diagnostics.window_days !== windowDays) {
      return null;
    }
    const series: LineSeriesOption[] = [];
    const componentValues = [
      diagnostics.velocity_mm_day,
      diagnostics.acceleration_mm_day2,
    ];
    const panelLabels = ["速度", "加速度"];
    const units = ["mm/日", "mm/日²"];
    componentValues.forEach((values, panelIndex) => {
      axes.forEach((axis, axisIndex) => {
        if (!visibleAxes[axisIndex]) {
          return;
        }
        series.push({
          name: `${axis} ${panelLabels[panelIndex]}`,
          type: "line",
          xAxisIndex: panelIndex,
          yAxisIndex: panelIndex,
          data: values.map((point) => point?.[axisIndex] ?? null),
          showSymbol: false,
          connectNulls: false,
          lineStyle: { width: 1.5, opacity: 0.8 },
          itemStyle: { color: axisColors[axisIndex] },
          tooltip: {
            valueFormatter: (value) =>
              typeof value === "number"
                ? `${value.toPrecision(4)} ${units[panelIndex]}`
                : "不可估计",
          },
        });
      });
    });
    series.push({
      name: "3D速度模（全部轴）",
      type: "line",
      data: diagnostics.speed_mm_day,
      showSymbol: false,
      connectNulls: false,
      lineStyle: { width: 2.1 },
      itemStyle: { color: "#34473c" },
      tooltip: {
        valueFormatter: (value) =>
          typeof value === "number"
            ? `${value.toPrecision(4)} mm/日`
            : "不可估计",
      },
    });
    series.push({
      name: "切向加速度（全部轴）",
      type: "line",
      xAxisIndex: 1,
      yAxisIndex: 1,
      data: diagnostics.tangential_acceleration_mm_day2,
      showSymbol: false,
      connectNulls: false,
      lineStyle: { width: 2.1, type: "dashed" },
      itemStyle: { color: "#34473c" },
      tooltip: {
        valueFormatter: (value) =>
          typeof value === "number"
            ? `${value.toPrecision(4)} mm/日²`
            : "低速或支撑不足",
      },
    });
    return {
      animation: false,
      grid: [
        { left: 82, right: 24, top: 36, height: 165 },
        { left: 82, right: 24, top: 280, height: 165 },
      ],
      tooltip: { trigger: "axis", confine: true },
      axisPointer: { link: [{ xAxisIndex: "all" }] },
      xAxis: [0, 1].map((gridIndex) => ({
        type: "category" as const,
        gridIndex,
        data: observation.dates,
        boundaryGap: false,
        axisLine: { lineStyle: { color: "#cbd4cc" } },
        axisLabel: { color: "#728077", fontSize: 10 },
        axisTick: { show: false },
      })),
      yAxis: [0, 1].map((gridIndex) => ({
        type: "value" as const,
        gridIndex,
        name: `${panelLabels[gridIndex]} / ${units[gridIndex]}`,
        axisLabel: {
          color: "#728077",
          fontSize: 10,
          formatter: (value: number) => {
            if (value === 0) {
              return "0";
            }
            if (Math.abs(value) < 0.01) {
              return value.toExponential(1);
            }
            return Number(value.toPrecision(3)).toString();
          },
        },
        splitLine: { lineStyle: { color: "#e8ede7" } },
        nameTextStyle: { color: "#728077", fontSize: 11 },
      })),
      dataZoom: [
        {
          id: "landslide-time-inside",
          type: "inside",
          xAxisIndex: [0, 1],
          filterMode: "none",
        },
        {
          id: "landslide-time-slider",
          type: "slider",
          xAxisIndex: [0, 1],
          filterMode: "none",
          bottom: 10,
          height: 22,
          borderColor: "#dce5dc",
          fillerColor: "rgba(20,125,114,.08)",
        },
      ],
      series,
    };
  }, [diagnostics, windowDays, visibleAxes, observation.dates]);

  return (
    <section className="ls-diagnostics" aria-labelledby="ls-diagnostics-title">
      <div className="ls-section-heading">
        <div>
          <p className="eyebrow">OBSERVATION-DERIVED MOTION</p>
          <h3 id="ls-diagnostics-title">速度与加速度辅助图</h3>
          <p className="ls-diagnostics-subtitle">
            沿着同一条记录，比较活动的起止、减速与再次活动。
          </p>
        </div>
        <div className="ls-diagnostics-controls">
          <label>
            估计窗口
            <select
              aria-label="辅助图估计窗口"
              value={windowDays}
              onChange={(event) =>
                setWindowDays(Number(event.target.value) as 31 | 61 | 91)
              }
            >
              <option value="31">31 日</option>
              <option value="61">61 日</option>
              <option value="91">91 日</option>
            </select>
          </label>
          <a
            href={`${endpoint}/diagnostics.png?window_days=${windowDays}`}
            download={`${observation.case_id}-diagnostics-${windowDays}d.png`}
          >
            <ArrowDownToLine size={14} />
            下载辅助图
          </a>
        </div>
      </div>
      <div className="ls-diagnostics-legend">
        <span>彩线：N / E / U（沿用上方轴开关）</span>
        <span>
          <i />
          深色实线：3D速度模
        </span>
        <span>
          <i className="dashed" />
          深色虚线：切向加速度
        </span>
      </div>
      {loading && (
        <p className="ls-loading">
          <LoaderCircle className="ls-spin" size={15} />
          正在从观测估计速度与加速度…
        </p>
      )}
      {error && (
        <div className="ls-error" role="alert">
          {error}
          <button onClick={() => setRetry((current) => current + 1)}>
            重试
          </button>
        </div>
      )}
      {chartOption && (
        <ReactECharts
          option={chartOption}
          replaceMerge={["series"]}
          onChartReady={(chart) => {
            chart.group = chartGroup;
            const observationOptions = observationChart.current
              ?.getEchartsInstance()
              .getOption();
            const zoom = (
              observationOptions?.dataZoom as
                | { start: number; end: number }[]
                | undefined
            )?.[0];
            if (zoom) {
              chart.dispatchAction(
                { type: "dataZoom", start: zoom.start, end: zoom.end },
                { silent: true },
              );
            }
          }}
          style={{ height: 530, width: "100%" }}
        />
      )}
      <div className="ls-diagnostics-notes">
        <p>
          <strong>仅由观测估计。</strong> 三轴使用{windowDays}
          日居中鲁棒二次拟合；端点、缺测日和支撑不足处留空。时间缩放与位移图联动，适合离线辨读，不是实时预测。
        </p>
        <p>
          3D速度模在低速时有噪声正偏；切向加速度表示沿运动方向的增速或减速，速度未超过拟合误差尺度3倍时留空（仅作显示门槛）。较长窗口更平滑，也会模糊短阶段与转折边界。
        </p>
        <span>
          下载为无阶段标签的多面板图，可用于后续盲标；此处不生成自动识别结果。
        </span>
      </div>
    </section>
  );
}

export default function LandslidePage() {
  const [datasets, setDatasets] = useState<Manifest[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [manifest, setManifest] = useState<Manifest | null>(null);
  const [listLoading, setListLoading] = useState(true);
  const [error, setError] = useState("");
  const [seed, setSeed] = useState("20261009");
  const [count, setCount] = useState("24");
  const [days, setDays] = useState("1095");
  const [startDate, setStartDate] = useState("2022-01-01");
  const [submitting, setSubmitting] = useState(false);
  const [scenario, setScenario] = useState<Scenario | "all">("all");
  const [page, setPage] = useState(0);
  const [pageSize, setPageSize] = useState(12);
  const [observations, setObservations] = useState<Record<string, Observation>>(
    {},
  );
  const [observationsLoading, setObservationsLoading] = useState(false);
  const [observationError, setObservationError] = useState("");
  const [caseId, setCaseId] = useState("");
  const [minimumSpan, setMinimumSpan] = useState(60);
  const [paddingPercent, setPaddingPercent] = useState(25);
  const [sharedScale, setSharedScale] = useState(false);
  const [visibleAxes, setVisibleAxes] = useState([true, true, true]);
  const [showRaw, setShowRaw] = useState(true);
  const [showMedian, setShowMedian] = useState(true);
  const [truthSelection, setTruthSelection] = useState("");
  const [truth, setTruth] = useState<Truth | null>(null);
  const [truthLoading, setTruthLoading] = useState(false);
  const [truthError, setTruthError] = useState("");
  const [reload, setReload] = useState(0);
  const chartRef = useRef<ReactECharts>(null);
  const detailRef = useRef<HTMLElement>(null);
  const selectionKey = `${selectedId}/${caseId}`;
  const showTruth = truthSelection === selectionKey;
  const chartGroup = useId();

  useEffect(() => {
    connect(chartGroup);
    return () => {
      disconnect(chartGroup);
    };
  }, [chartGroup]);

  useEffect(() => {
    const controller = new AbortController();
    setListLoading(true);
    request<Manifest[]>(basePath, { signal: controller.signal })
      .then((rows) => {
        if (controller.signal.aborted) {
          return;
        }
        setDatasets(rows);
        setSelectedId((current) => current || rows[0]?.dataset_id || "");
        setError("");
      })
      .catch((cause: Error) => {
        if (!controller.signal.aborted) {
          setError(cause.message);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setListLoading(false);
        }
      });
    return () => controller.abort();
  }, [reload]);

  useEffect(() => {
    setManifest(null);
    setObservations({});
    setScenario("all");
    setPage(0);
    setCaseId("");
    if (!selectedId) {
      return;
    }
    const controller = new AbortController();
    let timer: number | undefined;
    const poll = async () => {
      try {
        const next = await request<Manifest>(`${basePath}/${selectedId}`, {
          signal: controller.signal,
        });
        if (controller.signal.aborted) {
          return;
        }
        setManifest(next);
        setDatasets((current) => [
          next,
          ...current.filter((item) => item.dataset_id !== next.dataset_id),
        ]);
        if (next.status === "queued" || next.status === "running") {
          timer = window.setTimeout(poll, 1500);
        }
      } catch (cause) {
        if (!controller.signal.aborted) {
          setError((cause as Error).message);
        }
      }
    };
    void poll();
    return () => {
      controller.abort();
      window.clearTimeout(timer);
    };
  }, [selectedId, reload]);

  const filteredCases = useMemo(
    () =>
      manifest?.cases.filter(
        (item) => scenario === "all" || item.scenario === scenario,
      ) ?? [],
    [manifest, scenario],
  );
  const pageCount = Math.max(1, Math.ceil(filteredCases.length / pageSize));
  const pageCases = filteredCases.slice(page * pageSize, (page + 1) * pageSize);
  const pageCaseKey = pageCases.map((item) => item.case_id).join(",");

  useEffect(() => {
    const controller = new AbortController();
    const ids = pageCaseKey ? pageCaseKey.split(",") : [];
    setObservations({});
    setObservationError("");
    setCaseId((current) => (ids.includes(current) ? current : ids[0] || ""));
    if (!selectedId || ids.length === 0) {
      setObservationsLoading(false);
      return () => controller.abort();
    }
    setObservationsLoading(true);
    Promise.all(
      ids.map(async (id) => {
        const input = await request<Observation>(
          `${basePath}/${selectedId}/cases/${id}`,
          { signal: controller.signal },
        );
        return [id, input] as const;
      }),
    )
      .then((entries) => {
        if (!controller.signal.aborted) {
          setObservations(Object.fromEntries(entries));
        }
      })
      .catch((cause: Error) => {
        if (!controller.signal.aborted) {
          setObservationError(cause.message);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setObservationsLoading(false);
        }
      });
    return () => controller.abort();
  }, [selectedId, pageCaseKey, reload]);

  useEffect(() => {
    setTruthSelection("");
    setTruth(null);
    setTruthError("");
  }, [selectedId, caseId]);

  useEffect(() => {
    const controller = new AbortController();
    setTruth(null);
    setTruthError("");
    setTruthLoading(false);
    if (!showTruth || !selectedId || !caseId) {
      return () => controller.abort();
    }
    setTruthLoading(true);
    request<Truth>(`${basePath}/${selectedId}/cases/${caseId}/truth`, {
      signal: controller.signal,
    })
      .then((next) => {
        if (!controller.signal.aborted) {
          setTruth(next);
        }
      })
      .catch((cause: Error) => {
        if (!controller.signal.aborted) {
          setTruthError(cause.message);
        }
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setTruthLoading(false);
        }
      });
    return () => controller.abort();
  }, [selectedId, caseId, showTruth]);

  async function generate(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setSubmitting(true);
    setError("");
    try {
      const next = await request<Manifest>(basePath, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seed: Number(seed),
          count: Number(count),
          days: Number(days),
          start_date: startDate,
        }),
      });
      setDatasets((current) => [next, ...current]);
      setSelectedId(next.dataset_id);
    } catch (cause) {
      setError((cause as Error).message);
    } finally {
      setSubmitting(false);
    }
  }

  const selectedObservation = observations[caseId];
  const selectedCase = manifest?.cases.find((item) => item.case_id === caseId);
  const selectedMedian = useMemo(
    () =>
      selectedObservation
        ? rollingMedian(selectedObservation.displacement_mm)
        : [],
    [selectedObservation],
  );
  const batchExtent =
    manifest?.observed_extent_mm ?? ([0, 0] as [number, number]);
  const detailExtent = useMemo(() => {
    let extent: [number, number] = [0, 0];
    if (sharedScale) {
      extent = [...batchExtent];
    } else if (selectedObservation) {
      extent = observedExtent(selectedObservation);
    }
    if (showTruth && truth?.case_id === caseId) {
      for (const point of truth.latent_displacement_mm) {
        for (const value of point) {
          extent[0] = Math.min(extent[0], value);
          extent[1] = Math.max(extent[1], value);
        }
      }
    }
    return paddedExtent(extent, minimumSpan, paddingPercent);
  }, [
    selectedObservation,
    batchExtent,
    sharedScale,
    minimumSpan,
    paddingPercent,
    truth,
    showTruth,
    caseId,
  ]);

  const chartOption = useMemo<EChartsOption | null>(() => {
    if (!selectedObservation) {
      return null;
    }
    const series: LineSeriesOption[] = [];
    axes.forEach((axis, index) => {
      if (!visibleAxes[index]) {
        return;
      }
      if (showRaw) {
        series.push({
          name: `${axis} 原始`,
          type: "line",
          data: selectedObservation.displacement_mm.map(
            (point) => point?.[index] ?? null,
          ),
          showSymbol: false,
          connectNulls: false,
          lineStyle: { width: 1, opacity: 0.35 },
          itemStyle: { color: axisColors[index] },
          z: 1,
        });
      }
      if (showMedian) {
        series.push({
          name: `${axis} 31日中位数`,
          type: "line",
          data: selectedMedian.map((point) => point?.[index] ?? null),
          showSymbol: false,
          connectNulls: false,
          lineStyle: { width: 2.3 },
          itemStyle: { color: axisColors[index] },
          z: 3,
        });
      }
      if (showTruth && truth?.case_id === caseId) {
        series.push({
          name: `${axis} 潜在形变真值`,
          type: "line",
          data: truth.latent_displacement_mm.map((point) => point[index]),
          showSymbol: false,
          lineStyle: { width: 1.8, type: "dashed" },
          itemStyle: { color: axisColors[index] },
          z: 4,
        });
      }
    });
    return {
      animation: false,
      color: axisColors,
      grid: { left: 70, right: 24, top: 35, bottom: 85 },
      tooltip: {
        trigger: "axis",
        valueFormatter: (value) =>
          typeof value === "number" ? `${value.toFixed(2)} mm` : "缺测",
        confine: true,
      },
      xAxis: {
        type: "category",
        data: selectedObservation.dates,
        boundaryGap: false,
        axisLine: { lineStyle: { color: "#cbd4cc" } },
        axisLabel: { color: "#728077", fontSize: 11 },
        axisTick: { show: false },
      },
      yAxis: {
        type: "value",
        name: "位移 / mm",
        min: detailExtent[0],
        max: detailExtent[1],
        axisLabel: {
          color: "#728077",
          formatter: (value: number) => value.toFixed(0),
        },
        splitLine: { lineStyle: { color: "#e8ede7" } },
        nameTextStyle: { color: "#728077" },
      },
      dataZoom: [
        { id: "landslide-time-inside", type: "inside", filterMode: "none" },
        {
          id: "landslide-time-slider",
          type: "slider",
          filterMode: "none",
          bottom: 12,
          height: 24,
          borderColor: "#dce5dc",
          fillerColor: "rgba(20,125,114,.08)",
          dataBackground: { lineStyle: { color: "#8fa79b" } },
          selectedDataBackground: { lineStyle: { color: "#147d72" } },
        },
      ],
      series,
    };
  }, [
    selectedObservation,
    selectedMedian,
    visibleAxes,
    showRaw,
    showMedian,
    truth,
    showTruth,
    caseId,
    detailExtent,
  ]);

  let batchEmptyMessage = "从上方生成第一批数据，开始观察多年形变。";
  if (selectedId) {
    batchEmptyMessage = "正在读取批次…";
  } else if (listLoading) {
    batchEmptyMessage = "正在读取历史…";
  }
  let observationEmptyMessage = "点击上方案例卡片查看详细曲线";
  if (observationError) {
    observationEmptyMessage = "观测读取失败，请重试。";
  } else if (caseId) {
    observationEmptyMessage = "正在读取观测…";
  }

  return (
    <div className="ls-page">
      <section className="ls-intro">
        <div>
          <p className="eyebrow">
            LONG-TERM DISPLACEMENT / FIELD-INSPIRED SIMULATION
          </p>
          <h1>
            让长期形变<span>显现出来。</span>
          </h1>
          <p className="ls-intro-copy">
            在同一条多年记录中，观察蠕变、加速、减速、停滞与再活动的交替。结合原始位移和速度、加速度辅助图，辨认中段的变化与活动间歇。
          </p>
        </div>
        <div className="ls-intro-aside">
          <Mountain size={36} strokeWidth={1.15} />
          <span>合成实验数据</span>
          <small>机制启发 · 非真实测站记录</small>
          <a href="?view=docs&doc=landslide-design">
            研究依据与设计 <ArrowUpRight size={14} />
          </a>
        </div>
      </section>

      <form className="ls-generator" onSubmit={generate}>
        <div className="ls-generator-title">
          <span className="eyebrow">NEW BATCH</span>
          <strong>生成一批长期位移</strong>
        </div>
        <label>
          案例数量
          <input
            aria-label="案例数量"
            type="number"
            min="1"
            max="120"
            required
            value={count}
            onChange={(event) => setCount(event.target.value)}
          />
        </label>
        <label>
          观测天数
          <input
            aria-label="观测天数"
            type="number"
            min="730"
            max="2192"
            required
            value={days}
            onChange={(event) => setDays(event.target.value)}
          />
        </label>
        <label>
          开始日期
          <input
            aria-label="开始日期"
            type="date"
            required
            value={startDate}
            onChange={(event) => setStartDate(event.target.value)}
          />
        </label>
        <label>
          随机种子
          <input
            aria-label="随机种子"
            type="number"
            min="0"
            max="4294967295"
            required
            value={seed}
            onChange={(event) => setSeed(event.target.value)}
          />
        </label>
        <button className="ls-primary" disabled={submitting}>
          {submitting ? (
            <LoaderCircle className="ls-spin" size={16} />
          ) : (
            <Plus size={16} />
          )}
          {submitting ? "提交中" : "生成数据"}
        </button>
      </form>
      <p className="ls-generation-note">
        默认24例包含20条复合阶段记录与4条稳定对照；每条复合记录可在中段启动、结束或再次活动。
      </p>
      {error && (
        <div className="ls-error" role="alert">
          {error}
          <button onClick={() => setReload((value) => value + 1)}>
            重新读取
          </button>
        </div>
      )}

      <section className="ls-batch-bar">
        <label>
          <span className="eyebrow">BATCH ARCHIVE</span>
          <select
            aria-label="长期位移批次"
            value={selectedId}
            onChange={(event) => setSelectedId(event.target.value)}
          >
            {datasets.length === 0 && (
              <option value="">
                {listLoading ? "正在读取批次…" : "尚无长期位移批次"}
              </option>
            )}
            {datasets.map((item) => (
              <option key={item.dataset_id} value={item.dataset_id}>
                {item.dataset_id} · {item.request.count}例 ·{" "}
                {statusLabels[item.status]}
              </option>
            ))}
          </select>
        </label>
        {manifest && (
          <div className="ls-batch-stats">
            <div>
              <strong>
                {manifest.generated_cases}
                <small> / {manifest.request.count}</small>
              </strong>
              <span>已生成案例</span>
            </div>
            <div>
              <strong>
                {(manifest.request.days / 365.25).toFixed(1)}
                <small> 年</small>
              </strong>
              <span>{manifest.request.days} 个日历日</span>
            </div>
            <div>
              <strong>
                {Object.keys(manifest.type_counts).length}
                <small> 类</small>
              </strong>
              <span>{statusLabels[manifest.status]}</span>
            </div>
          </div>
        )}
      </section>
      {!manifest && <div className="ls-empty">{batchEmptyMessage}</div>}
      {manifest && (
        <>
          {(manifest.status === "queued" || manifest.status === "running") && (
            <div className="ls-progress">
              <LoaderCircle className="ls-spin" size={15} />
              正在生成 {manifest.generated_cases} / {manifest.request.count}{" "}
              例，曲线将自动更新。
              <progress
                max={manifest.request.count}
                value={manifest.generated_cases}
              />
            </div>
          )}
          {manifest.status === "failed" && (
            <div className="ls-error" role="alert">
              {manifest.error || "批次生成失败"}
            </div>
          )}
          <section className="ls-scale-panel" aria-label="纵轴显示设置">
            <div>
              <strong>看清趋势的尺度</strong>
              <p>保留零点及全部观测极值；缩放时间时纵轴保持稳定。</p>
            </div>
            <label>
              最小数据跨度
              <input
                aria-label="最小数据跨度 mm"
                type="number"
                min="10"
                max="100000"
                step="10"
                value={minimumSpan}
                onChange={(event) =>
                  setMinimumSpan(
                    Math.max(
                      10,
                      Math.min(100000, Number(event.target.value) || 10),
                    ),
                  )
                }
              />
              <span>mm</span>
            </label>
            <label>
              上下各留白
              <input
                aria-label="上下各留白百分比"
                type="number"
                min="0"
                max="100"
                step="5"
                value={paddingPercent}
                onChange={(event) =>
                  setPaddingPercent(
                    Math.max(0, Math.min(100, Number(event.target.value))),
                  )
                }
              />
              <span>%</span>
            </label>
            <label className="ls-scale-toggle">
              <input
                type="checkbox"
                checked={sharedScale}
                onChange={(event) => setSharedScale(event.target.checked)}
              />
              批次统一纵轴
            </label>
          </section>

          <section className="ls-overview" aria-labelledby="ls-overview-title">
            <div className="ls-section-heading">
              <div>
                <p className="eyebrow">OBSERVATION ATLAS</p>
                <h2 id="ls-overview-title">批量曲线总览</h2>
              </div>
              <div className="ls-legend">
                {axes.map((axis, index) => (
                  <span key={axis}>
                    <i style={{ background: axisColors[index] }} />
                    {axis}
                  </span>
                ))}
                <small>淡线：观测 · 实线：31日中位数</small>
              </div>
            </div>
            <p className="ls-overview-note">
              下方标签表示生成场景；同一条复合记录内包含多个阶段，不以整条记录的单一标签代替时间段判断。
            </p>
            <div className="ls-filter-row">
              <div
                className="ls-filters"
                role="group"
                aria-label="生成场景筛选"
              >
                <button
                  className={scenario === "all" ? "active" : ""}
                  onClick={() => {
                    setScenario("all");
                    setPage(0);
                  }}
                >
                  全部 <span>{manifest.cases.length}</span>
                </button>
                {(Object.entries(scenarioLabels) as [Scenario, string][])
                  .filter(([key]) => (manifest.type_counts[key] ?? 0) > 0)
                  .map(([key, label]) => (
                    <button
                      className={scenario === key ? "active" : ""}
                      key={key}
                      onClick={() => {
                        setScenario(key);
                        setPage(0);
                      }}
                    >
                      {label}
                      <span>{manifest.type_counts[key] ?? 0}</span>
                    </button>
                  ))}
              </div>
              <span className="ls-scale-badge">
                {sharedScale ? "批次统一尺度" : "各例独立尺度"}
              </span>
            </div>
            {observationsLoading && (
              <p className="ls-loading">
                <LoaderCircle className="ls-spin" size={16} />
                正在读取本页 {pageCases.length} 条观测序列…
              </p>
            )}
            {observationError && (
              <div className="ls-error" role="alert">
                {observationError}
                <button onClick={() => setReload((value) => value + 1)}>
                  重试
                </button>
              </div>
            )}
            <div className="ls-cards">
              {pageCases.map((item) => {
                const observation = observations[item.case_id];
                let dataExtent: [number, number] = [0, 0];
                if (sharedScale) {
                  dataExtent = batchExtent;
                } else if (observation) {
                  dataExtent = observedExtent(observation);
                }
                const extent = paddedExtent(
                  dataExtent,
                  minimumSpan,
                  paddingPercent,
                );
                return (
                  <button
                    className={`ls-card ${caseId === item.case_id ? "selected" : ""}`}
                    key={item.case_id}
                    onClick={() => {
                      setCaseId(item.case_id);
                      detailRef.current?.scrollIntoView({
                        behavior: "smooth",
                        block: "start",
                      });
                    }}
                    aria-pressed={caseId === item.case_id}
                  >
                    <div className="ls-card-heading">
                      <strong>{item.case_id.replace("case_", "#")}</strong>
                      <span>{scenarioLabels[item.scenario]}</span>
                      <ArrowUpRight size={15} />
                    </div>
                    {observation ? (
                      <OverviewCurve
                        observation={observation}
                        extent={extent}
                      />
                    ) : (
                      <div className="ls-mini-placeholder">
                        {observationError ? "读取失败" : "读取观测中"}
                      </div>
                    )}
                    <div className="ls-card-scale">
                      <span>
                        {extent[0].toFixed(0)} ~ {extent[1].toFixed(0)} mm
                      </span>
                      <span>缺测 {item.missing_days} 日</span>
                    </div>
                    <div className="ls-card-dates">
                      <span>
                        {observation?.dates[0] ?? manifest.request.start_date}
                      </span>
                      <span>{observation?.dates.at(-1) ?? ""}</span>
                    </div>
                  </button>
                );
              })}
            </div>
            {pageCases.length === 0 && (
              <div className="ls-empty">
                {manifest.status === "running" || manifest.status === "queued"
                  ? "等待首批案例完成…"
                  : "当前筛选下没有案例"}
              </div>
            )}
            <div className="ls-pagination">
              <span>共 {filteredCases.length} 例 · 只读取当前页观测</span>
              <label>
                每页
                <select
                  aria-label="每页案例数"
                  value={pageSize}
                  onChange={(event) => {
                    setPageSize(Number(event.target.value));
                    setPage(0);
                  }}
                >
                  <option value="12">12 例</option>
                  <option value="24">24 例</option>
                </select>
              </label>
              <button
                aria-label="上一页"
                disabled={page === 0}
                onClick={() => setPage((current) => current - 1)}
              >
                <ChevronLeft size={16} />
              </button>
              <span>
                {page + 1} / {pageCount}
              </span>
              <button
                aria-label="下一页"
                disabled={page + 1 >= pageCount}
                onClick={() => setPage((current) => current + 1)}
              >
                <ChevronRight size={16} />
              </button>
            </div>
          </section>

          <section
            className="ls-detail"
            ref={detailRef}
            aria-labelledby="ls-detail-title"
          >
            <div className="ls-section-heading">
              <div>
                <p className="eyebrow">DISPLACEMENT RECORD</p>
                <h2 id="ls-detail-title">
                  {caseId || "选择案例"}
                  <span>
                    {selectedCase ? scenarioLabels[selectedCase.scenario] : ""}
                  </span>
                </h2>
              </div>
              {selectedObservation && (
                <div className="ls-downloads">
                  <a
                    href={`${basePath}/${selectedId}/cases/${caseId}/observations.csv`}
                    download
                  >
                    <ArrowDownToLine size={14} />
                    观测 CSV
                  </a>
                  <a
                    href={`${basePath}/${selectedId}/cases/${caseId}`}
                    download={`${caseId}.json`}
                  >
                    <ArrowDownToLine size={14} />
                    观测 JSON
                  </a>
                </div>
              )}
            </div>
            <div className="ls-detail-toolbar">
              <div className="ls-axis-toggles">
                {axes.map((axis, index) => (
                  <label key={axis}>
                    <input
                      type="checkbox"
                      checked={visibleAxes[index]}
                      onChange={() =>
                        setVisibleAxes((current) =>
                          current.map((value, position) =>
                            index === position ? !value : value,
                          ),
                        )
                      }
                    />
                    <i style={{ background: axisColors[index] }} />
                    {axis}
                  </label>
                ))}
              </div>
              <label>
                <input
                  type="checkbox"
                  checked={showRaw}
                  onChange={(event) => setShowRaw(event.target.checked)}
                />
                原始观测
              </label>
              <label>
                <input
                  type="checkbox"
                  checked={showMedian}
                  onChange={(event) => setShowMedian(event.target.checked)}
                />
                31日中位数
              </label>
              <button
                onClick={() =>
                  chartRef.current
                    ?.getEchartsInstance()
                    .dispatchAction({ type: "dataZoom", start: 0, end: 100 })
                }
              >
                <RotateCcw size={14} />
                完整时间范围
              </button>
            </div>
            {chartOption ? (
              <ReactECharts
                key={`displacement:${selectionKey}`}
                ref={chartRef}
                option={chartOption}
                replaceMerge={["series"]}
                onChartReady={(chart) => {
                  chart.group = chartGroup;
                }}
                style={{ height: 410, width: "100%" }}
              />
            ) : (
              <div className="ls-empty">{observationEmptyMessage}</div>
            )}
            <div className="ls-chart-note">
              <span>
                {selectedObservation
                  ? `${selectedObservation.dates[0]} — ${selectedObservation.dates.at(-1)}`
                  : "日尺度观测"}{" "}
                · {sharedScale ? "批次统一纵轴" : "当前案例纵轴"}
                {showTruth && truth ? "（含潜在形变范围）" : ""}
              </span>
              <span>
                中位数使用前后各15日；端点窗口缩短。缺测日断线，非预测结果。
              </span>
            </div>
            {selectedObservation && (
              <DiagnosticsPanel
                key={`diagnostics:${selectionKey}`}
                observation={selectedObservation}
                datasetId={selectedId}
                visibleAxes={visibleAxes}
                chartGroup={chartGroup}
                observationChart={chartRef}
              />
            )}
            <div className="ls-truth-access">
              <div>
                <strong>仿真真值核对</strong>
                <p>按需查看潜在形变与阶段标签，用于解释生成机制。</p>
              </div>
              <button
                disabled={!selectedObservation}
                aria-pressed={showTruth}
                onClick={() => setTruthSelection(showTruth ? "" : selectionKey)}
              >
                {showTruth ? "隐藏真值" : "查看真值"}
              </button>
            </div>
            {truthLoading && (
              <p className="ls-loading">
                <LoaderCircle className="ls-spin" size={15} />
                正在读取真值…
              </p>
            )}
            {truthError && (
              <div className="ls-error" role="alert">
                {truthError}
              </div>
            )}
            {showTruth && truth?.case_id === caseId && selectedObservation && (
              <div className="ls-truth-panel">
                <p>
                  虚线为潜在形变真值，不属于观测输入。以下边界来自生成机制，不代表从观测中可精确识别的起止时间，也不是检测结果。
                </p>
                <div
                  className="ls-phase-timeline"
                  aria-label="生成机制阶段时间带"
                >
                  {truth.phases.map((phase, index) => (
                    <div
                      key={`${phase.kind}-${index}`}
                      title={`${phaseLabels[phase.kind] ?? phase.kind} · ${selectedObservation.dates[phase.start_index]} — ${selectedObservation.dates[phase.end_index]}`}
                      style={{
                        left: `${(phase.start_index / selectedObservation.dates.length) * 100}%`,
                        width: `${((phase.end_index - phase.start_index + 1) / selectedObservation.dates.length) * 100}%`,
                        background: phaseColors[phase.kind] ?? "#b0bca6",
                      }}
                    >
                      <span>{phaseLabels[phase.kind] ?? phase.kind}</span>
                    </div>
                  ))}
                </div>
                <div className="ls-phase-timeline-dates">
                  <span>{selectedObservation.dates[0]}</span>
                  <span>{selectedObservation.dates.at(-1)}</span>
                </div>
                <div className="ls-phases">
                  {truth.phases.map((phase, index) => (
                    <div key={`${phase.kind}-${index}`}>
                      <span>
                        {selectedObservation.dates[phase.start_index]} —{" "}
                        {selectedObservation.dates[phase.end_index]}
                      </span>
                      <b className="ls-phase-name">
                        <i
                          style={{
                            background: phaseColors[phase.kind] ?? "#b0bca6",
                          }}
                        />
                        {phaseLabels[phase.kind] ?? phase.kind}
                      </b>
                      <strong>{phase.description}</strong>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        </>
      )}
      <footer className="ls-footer">
        <span>GNSS LAB / LANDSLIDE SERIES</span>
        <span>从机制出发，以观测为准。</span>
      </footer>
    </div>
  );
}
