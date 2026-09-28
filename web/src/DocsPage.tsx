import { useEffect, useMemo, useRef, useState } from "react";
import { ArrowLeft, ArrowRight, ArrowUp, BookOpenText, Check, Copy, Download, ExternalLink, List, LoaderCircle, Printer, Search, X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";

type DocumentEntry = { slug: string; title: string; file: string };
type Heading = { id: string; title: string; level: number };
type MarkdownNode = { type: string; value?: string; children?: MarkdownNode[]; data?: { hProperties?: Record<string, unknown> } };
const groups = [
  { label: "阅读起点", slugs: ["experimental-handbook", "research-design", "experiment-audit"] },
  { label: "冻结实验协议", slugs: ["p1-normal-model", "data-and-reproducibility", "planned-methods", "p4-pilot-evaluator", "p5-numerical-baselines", "p6-visual-baseline"] },
  { label: "比较与后续设计", slugs: ["p8a-visual-range-context", "p7b-error-analysis", "independent-confirmation", "p7b-candidate-review", "versioned-comparison", "p7-design"] },
];
function headingText(node: MarkdownNode): string { return node.value ?? node.children?.map(headingText).join("") ?? ""; }
// Derive unique anchors from the parsed tree so inline formatting does not break links.
function headingAnchors() {
  return (tree: MarkdownNode) => {
    const counts = new Map<string, number>();
    function visit(node: MarkdownNode) {
      if (node.type === "heading") {
        const base = headingText(node).trim().toLowerCase().replace(/\s+/g, "-").replace(/[^\p{L}\p{N}-]/gu, "") || "section";
        const count = counts.get(base) ?? 0;
        counts.set(base, count + 1);
        node.data = { ...node.data, hProperties: { ...node.data?.hProperties, id: count ? `${base}-${count}` : base } };
      }
      node.children?.forEach(visit);
    }
    visit(tree);
  };
}
function currentSlug() { return new URLSearchParams(location.search).get("doc") ?? "experimental-handbook"; }
function currentAnchor() { try { return decodeURIComponent(location.hash.slice(1)); } catch { return ""; } }
function scrollToAnchor(id: string, smooth = true) {
  const target = document.getElementById(id || "reading-paper");
  target?.scrollIntoView({ behavior: smooth && !matchMedia("(prefers-reduced-motion: reduce)").matches ? "smooth" : "auto", block: "start" });
}
async function readResponse(response: Response): Promise<string> {
  if (response.status === 404) throw new Error("未找到文档。若刚更新项目，请重启本地后端服务后重试。");
  if (!response.ok) throw new Error(`文档读取失败（${response.status}）。请检查本地服务后重试。`);
  return response.text();
}

export default function DocsPage() {
  const [documents, setDocuments] = useState<DocumentEntry[]>([]);
  const [selected, setSelected] = useState(currentSlug);
  const [markdown, setMarkdown] = useState("");
  const [cache, setCache] = useState<Record<string, string>>({});
  const cacheRef = useRef(cache);
  cacheRef.current = cache;
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [retry, setRetry] = useState(0);
  const [query, setQuery] = useState("");
  const [headings, setHeadings] = useState<Heading[]>([]);
  const [active, setActive] = useState("");
  const [progress, setProgress] = useState(0);
  const [notice, setNotice] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const [searching, setSearching] = useState(false);
  const paper = useRef<HTMLElement>(null);
  const search = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetch("/api/docs", { signal: controller.signal, cache: "no-store" })
      .then(async response => JSON.parse(await readResponse(response)) as DocumentEntry[])
      .then(items => { if (!controller.signal.aborted) setDocuments(items); })
      .catch(cause => { if (!controller.signal.aborted) { setError(String(cause.message)); setLoading(false); } });
    return () => controller.abort();
  }, [retry]);

  useEffect(() => {
    if (!documents.length) return;
    const controller = new AbortController();
    setLoading(true); setError(""); setHeadings([]); setProgress(0); setActive("");
    if (!documents.some(item => item.slug === selected)) {
      setError("这个文档地址不存在。请从目录选择文档，或返回实验方法手册。"); setLoading(false); return;
    }
    fetch(`/api/docs/${encodeURIComponent(selected)}`, { signal: controller.signal, cache: "no-store" })
      .then(readResponse).then(body => {
        if (controller.signal.aborted) return;
        setMarkdown(body); setCache(old => ({ ...old, [selected]: body })); setLoading(false);
      }).catch(cause => { if (!controller.signal.aborted) { setError(String(cause.message)); setLoading(false); } });
    return () => controller.abort();
  }, [selected, documents, retry]);

  // Search only public research documents, never datasets or labels.
  useEffect(() => {
    if (!query.trim()) { setSearching(false); return; }
    const controller = new AbortController();
    setSearching(true);
    const timer = window.setTimeout(async () => {
      await Promise.allSettled(documents.filter(item => !(item.slug in cacheRef.current)).map(async item => {
        const body = await readResponse(await fetch(`/api/docs/${item.slug}`, { signal: controller.signal }));
        if (!controller.signal.aborted) setCache(old => ({ ...old, [item.slug]: body }));
      }));
      if (!controller.signal.aborted) setSearching(false);
    }, 200);
    return () => { clearTimeout(timer); controller.abort(); };
  }, [query, documents]);

  useEffect(() => {
    const restore = () => { setSelected(currentSlug()); setMenuOpen(false); scrollToAnchor(currentAnchor(), false); };
    window.addEventListener("popstate", restore); window.addEventListener("hashchange", restore);
    return () => { window.removeEventListener("popstate", restore); window.removeEventListener("hashchange", restore); };
  }, []);

  useEffect(() => {
    if (loading || error) return;
    const nodes = Array.from(paper.current?.querySelectorAll<HTMLElement>(".docs-prose h2, .docs-prose h3") ?? []);
    setHeadings(nodes.map(node => ({ id: node.id, title: node.textContent ?? "", level: Number(node.tagName.slice(1)) })));
    const frame = requestAnimationFrame(() => { if (currentAnchor()) scrollToAnchor(currentAnchor(), false); });
    let pending = 0;
    const update = () => {
      pending = 0;
      const visible = nodes.filter(node => node.getBoundingClientRect().top <= 160);
      setActive(visible.at(-1)?.id ?? nodes[0]?.id ?? "");
      const rect = paper.current?.getBoundingClientRect();
      if (rect) setProgress(Math.max(0, Math.min(100, Math.round((140 - rect.top) / Math.max(1, rect.height - innerHeight + 180) * 100))));
    };
    const schedule = () => { if (!pending) pending = requestAnimationFrame(update); };
    window.addEventListener("scroll", schedule, { passive: true }); window.addEventListener("resize", schedule); update();
    return () => { cancelAnimationFrame(frame); cancelAnimationFrame(pending); window.removeEventListener("scroll", schedule); window.removeEventListener("resize", schedule); };
  }, [markdown, loading, error]);

  useEffect(() => { if (!notice) return; const timer = setTimeout(() => setNotice(""), 3500); return () => clearTimeout(timer); }, [notice]);
  const ordered = useMemo(() => groups.flatMap(group => group.slugs.map(slug => documents.find(item => item.slug === slug)).filter((item): item is DocumentEntry => !!item)), [documents]);
  const index = ordered.findIndex(item => item.slug === selected);
  const title = documents.find(item => item.slug === selected)?.title ?? "实验文档";
  const terms = query.trim().toLocaleLowerCase();
  const matches = (item: DocumentEntry) => !terms || `${item.title}\n${cache[item.slug] ?? ""}`.toLocaleLowerCase().includes(terms);
  const results = ordered.filter(matches);
  const readingMinutes = Math.max(1, Math.ceil(markdown.replace(/\s/g, "").length / 550));

  function navigate(slug: string, anchor = "") {
    const url = new URL(location.href);
    url.searchParams.set("view", "docs"); url.searchParams.set("doc", slug); url.hash = anchor;
    history.pushState(null, "", url); setSelected(slug); setMenuOpen(false); setNotice("");
    if (selected === slug) scrollToAnchor(anchor); else window.scrollTo({ top: 0, behavior: "auto" });
  }
  async function copyLink() {
    const url = new URL(location.href); url.searchParams.set("view", "docs"); url.searchParams.set("doc", selected);
    try { await navigator.clipboard.writeText(url.href); setNotice("已复制文档链接"); }
    catch { setNotice("复制不可用，请从浏览器地址栏手动复制。"); }
  }
  function download() {
    const url = URL.createObjectURL(new Blob([markdown], { type: "text/markdown;charset=utf-8" }));
    const anchor = document.createElement("a"); anchor.href = url;
    anchor.download = documents.find(item => item.slug === selected)?.file ?? `${selected}.md`;
    anchor.click(); setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  return (
    <section className="docs-page" aria-label="实验文档阅读器">
      <a className="docs-skip" href="#reading-paper">跳到正文</a>
      <div className="docs-hero"><div><p className="docs-eyebrow">GNSS LAB / RESEARCH LIBRARY</p><h1>实验方法与研究记录<span>从任务定义，到可重复的证据。</span></h1></div><div className="docs-edition"><span>METHODS · PROTOCOLS · FINDINGS</span><strong>开发实验 P1—P7b</strong><small>复核实验已完成 · 独立确认未启动</small></div></div>
      <div className="docs-mobile-bar"><button onClick={() => setMenuOpen(!menuOpen)} aria-expanded={menuOpen} aria-controls="chapter-navigation"><List size={17} /> 文档目录</button><span>{title}</span></div>
      <div className="docs-layout">
        <nav id="chapter-navigation" className={`docs-chapters ${menuOpen ? "is-open" : ""}`} aria-label="选择文档">
          <label className="docs-search"><Search size={16} /><input ref={search} aria-label="搜索文档全文" placeholder="搜索术语、指标、协议…" value={query} onChange={event => setQuery(event.target.value)} onKeyDown={event => { if (event.key === "Escape") setQuery(""); }} />{query && <button aria-label="清除搜索" onClick={() => { setQuery(""); search.current?.focus(); }}><X size={14} /></button>}</label>
          {terms && <p className="docs-search-count" role="status">{results.length} 篇匹配{searching ? " · 正在检索正文" : documents.some(item => !(item.slug in cache)) ? " · 部分正文暂未加载" : ""}</p>}
          {groups.map(group => <div className="docs-group" key={group.label}><p>{group.label}</p>{ordered.filter(item => group.slugs.includes(item.slug) && matches(item)).map(item => <button type="button" key={item.slug} className={`docs-chapter ${selected === item.slug ? "selected" : ""}`} aria-current={selected === item.slug ? "page" : undefined} onClick={() => navigate(item.slug)}><span className="docs-chapter-number">{String(ordered.indexOf(item) + 1).padStart(2, "0")}</span><span>{item.title}{terms && !item.title.toLocaleLowerCase().includes(terms) && <small>正文包含「{query.trim()}」</small>}</span>{selected === item.slug && <span className="docs-current-dot" />}</button>)}</div>)}
          {terms && !results.length && !searching && <div className="docs-empty">没有匹配文档。试试“FAR”“活动区间”或“复核”。</div>}
          <div className="docs-rail-note"><BookOpenText size={17} /><span>首次阅读建议从方法手册开始。历史结果与待执行设计分开标注。</span></div>
        </nav>
        <article id="reading-paper" ref={paper} tabIndex={-1} className="docs-paper" aria-label={title} aria-busy={loading}>
          <div className="docs-reading-line" aria-hidden="true"><span style={{ width: `${progress}%` }} /></div>
          {loading ? <div className="docs-state" role="status"><LoaderCircle className="spin" size={24} /> 正在读取文档…</div> : error ? <div className="docs-state docs-error" role="alert"><p>{error}</p><button onClick={() => setRetry(value => value + 1)}>重试</button><button onClick={() => navigate("experimental-handbook")}>返回方法手册</button></div> : <>
            <div className="docs-paper-topline"><span>RESEARCH NOTE {String(index + 1).padStart(2, "0")}</span><span>约 {readingMinutes} 分钟阅读 · {headings.filter(item => item.level === 2).length} 节</span></div>
            <div className="docs-tools" aria-label="文档操作"><span>同源文档 · 可复现记录</span><div><button onClick={copyLink} title="复制当前文档与章节链接"><Copy size={15} /> 链接</button><button onClick={download} title="下载当前 Markdown 原稿"><Download size={15} /> 原稿</button><button onClick={() => window.print()} title="打印或另存为 PDF"><Printer size={15} /> 打印</button></div></div>
            <details className="docs-mobile-contents"><summary>本篇目录 · {headings.filter(item => item.level === 2).length} 节</summary>{headings.filter(item => item.level === 2).map(item => <button key={item.id} onClick={event => { navigate(selected, item.id); event.currentTarget.closest("details")?.removeAttribute("open"); }}>{item.title}</button>)}</details>
            <div className="docs-prose"><ReactMarkdown remarkPlugins={[remarkGfm, remarkMath, headingAnchors]} rehypePlugins={[rehypeKatex]} components={{
              table: ({ children }) => <div className="docs-table-scroll" tabIndex={0} role="region" aria-label="数据表格，可横向滚动"><table>{children}</table></div>,
              a: ({ href, children }) => {
                if (href?.startsWith("#")) return <a href={href} onClick={event => { event.preventDefault(); navigate(selected, decodeURIComponent(href.slice(1))); }}>{children}</a>;
                if (href && /^https?:\/\//.test(href)) return <a href={href} target="_blank" rel="noopener noreferrer">{children}<ExternalLink size={11} className="docs-external" aria-label="在新标签页打开" /></a>;
                const [path, anchor = ""] = (href ?? "").split("#");
                const target = documents.find(item => item.file === path.split("/").at(-1));
                if (target) return <a href={`?view=docs&doc=${target.slug}${anchor ? `#${anchor}` : ""}`} onClick={event => { if (event.ctrlKey || event.metaKey || event.shiftKey || event.altKey) return; event.preventDefault(); navigate(target.slug, decodeURIComponent(anchor)); }}>{children}</a>;
                return <span className="docs-source-ref" title={`项目本地文件：${href ?? ""}`}>{children}<small>本地来源</small></span>;
              },
            }}>{markdown}</ReactMarkdown></div>
            <div className="docs-page-turn">{index > 0 ? <button onClick={() => navigate(ordered[index - 1].slug)}><ArrowLeft size={18} /><span><small>上一篇</small>{ordered[index - 1].title}</span></button> : <span />}{index < ordered.length - 1 && <button onClick={() => navigate(ordered[index + 1].slug)}><span><small>下一篇</small>{ordered[index + 1].title}</span><ArrowRight size={18} /></button>}</div>
            <footer className="docs-paper-footer"><span>GNSS LAB · 受控合成实验</span><button onClick={() => scrollToAnchor("")}><ArrowUp size={14} /> 回到篇首</button></footer>
          </>}
        </article>
        <nav className="docs-contents" aria-label="本篇目录"><div className="docs-toc-label">ON THIS PAGE <span>{progress}%</span></div><div className="docs-toc-scroll">{headings.map(item => <button className={`${active === item.id ? "active" : ""} level-${item.level}`} key={item.id} aria-current={active === item.id ? "location" : undefined} onClick={() => navigate(selected, item.id)}>{item.title}</button>)}</div><button className="docs-back-top" onClick={() => scrollToAnchor("")}><ArrowUp size={14} /> 回到篇首</button></nav>
      </div>
      <div className={`docs-notice ${notice ? "visible" : ""}`} role="status">{notice && <><Check size={16} />{notice}</>}</div>
    </section>
  );
}
