import { useEffect, useMemo, useState } from "react";
import { BookOpenText, ChevronRight, FileText, LoaderCircle } from "lucide-react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

type DocumentEntry = { slug: string; title: string };
type Heading = { id: string; title: string };

function headingId(value: string): string {
  return value.trim().toLowerCase().replace(/\s+/g, "-").replace(/[^\p{L}\p{N}-]/gu, "");
}

function headingsFrom(markdown: string): Heading[] {
  return [...markdown.matchAll(/^## (.+)$/gm)].map((match) => ({
    title: match[1].trim(),
    id: headingId(match[1]),
  }));
}

async function readResponse(response: Response): Promise<string> {
  if (response.status === 404) {
    throw new Error("文档接口返回 404。请重启本地后端服务，再刷新页面。");
  }
  if (!response.ok) throw new Error(`文档读取失败 (${response.status})`);
  return response.text();
}

export default function DocsPage() {
  const [documents, setDocuments] = useState<DocumentEntry[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [markdown, setMarkdown] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/docs", { signal: controller.signal, cache: "no-store" })
      .then(async (response) => JSON.parse(await readResponse(response)) as DocumentEntry[])
      .then((items) => {
        setDocuments(items);
        setSelected((current) => current ?? items[0]?.slug ?? null);
      })
      .catch((cause) => {
        if (!controller.signal.aborted) {
          setError((cause as Error).message);
          setLoading(false);
        }
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    if (!selected) return;
    const controller = new AbortController();
    setLoading(true);
    setError("");
    fetch(`/api/docs/${encodeURIComponent(selected)}`, {
      signal: controller.signal,
      cache: "no-store",
    })
      .then(readResponse)
      .then((body) => {
        setMarkdown(body);
        setLoading(false);
      })
      .catch((cause) => {
        if (!controller.signal.aborted) {
          setError((cause as Error).message);
          setLoading(false);
        }
      });
    return () => controller.abort();
  }, [selected]);

  const headings = useMemo(() => headingsFrom(markdown), [markdown]);
  const title = documents.find((item) => item.slug === selected)?.title ?? "实验设计";

  function jumpTo(id: string) {
    document.getElementById(id)?.scrollIntoView({ behavior: "smooth", block: "start" });
  }

  return (
    <section className="docs-page" aria-label="实验设计文档">
      <div className="docs-hero">
        <div>
          <p className="eyebrow">FIELD NOTES / SIMULATION LAB</p>
          <h1>实验设计手册</h1>
          <p>从研究问题走到生成公式，再核对数据与复现流程。</p>
        </div>
        <span className="docs-stage">05 / P5 已验收</span>
      </div>

      <div className="docs-layout">
        <nav className="docs-chapters" aria-label="选择文档">
          <div className="docs-rail-label"><BookOpenText size={17} /> 阅读章节</div>
          {documents.map((item, index) => (
            <button
              type="button"
              key={item.slug}
              className={`docs-chapter ${selected === item.slug ? "selected" : ""}`}
              aria-current={selected === item.slug ? "page" : undefined}
              onClick={() => {
                setSelected(item.slug);
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
            >
              <span className="docs-chapter-number">{String(index + 1).padStart(2, "0")}</span>
              <span>{item.title}</span>
              <ChevronRight size={15} />
            </button>
          ))}
          <div className="docs-rail-note">
            <FileText size={16} /> 文档随实验代码维护；“未实现”章节仅说明研究路线。
          </div>
        </nav>

        <article className="docs-paper" aria-label={title}>
          {loading ? (
            <div className="docs-state"><LoaderCircle className="spin" size={24} /> 正在读取文档…</div>
          ) : error ? (
            <div className="docs-state docs-error" role="alert">{error}</div>
          ) : (
            <>
              <div className="docs-paper-topline"><span>GNSS SIMULATION LAB</span><span>{selected === "p1-normal-model" ? "FROZEN BACKGROUND · P1" : selected === "planned-methods" ? "EVENT PROTOCOL · P2/P3" : selected === "p4-pilot-evaluator" ? "PILOT & EVALUATOR · P4" : selected === "p5-numerical-baselines" ? "NUMERICAL BASELINES · P5" : "RESEARCH NOTE"}</span></div>
              <details className="docs-mobile-contents">
                <summary>本篇目录</summary>
                {headings.map((item) => <button key={item.id} onClick={() => jumpTo(item.id)}>{item.title}</button>)}
              </details>
              <div className="docs-prose">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                  components={{
                    h2: ({ children }) => <h2 id={headingId(String(children))}>{children}</h2>,
                  }}
                >
                  {markdown}
                </ReactMarkdown>
              </div>
              <div className="docs-paper-footer">实验文档 · 以当前代码与本地数据契约为准</div>
            </>
          )}
        </article>

        <nav className="docs-contents" aria-label="本篇目录">
          <p>本篇目录</p>
          {headings.map((item) => <button key={item.id} onClick={() => jumpTo(item.id)}>{item.title}</button>)}
        </nav>
      </div>
    </section>
  );
}
