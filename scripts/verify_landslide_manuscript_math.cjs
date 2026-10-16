// Validate the actual Markdown math with the repository's installed KaTeX renderer.
const fs = require('node:fs');
const path = require('node:path');
const crypto = require('node:crypto');
const katex = require('../web/node_modules/katex');

const manuscript = path.resolve(process.argv[2]);
const output = path.resolve(process.argv[3]);
const bytes = fs.readFileSync(manuscript);
const markdown = bytes.toString('utf8');
// Markdown code fences can contain shell variables such as $env. Keep their
// line positions while excluding literal code from mathematical delimiters.
const formulaMarkdown = markdown.replace(/^```[^\n]*\n[\s\S]*?^```[^\n]*(?=\n|$)/gm,
  block => block.replace(/[^\r\n]/g, ' '));
const formulaPattern = /\$\$([\s\S]*?)\$\$|(?<![\\$])\$(?!\$)([^$\n]+)\$(?!\$)/g;
const formulas = [];
for (const match of formulaMarkdown.matchAll(formulaPattern)) {
  const displayMode = match[1] !== undefined;
  const latex = (displayMode ? match[1] : match[2]).trim();
  const html = katex.renderToString(latex, { displayMode, throwOnError: true, strict: 'error' });
  const line = markdown.slice(0, match.index).split('\n').length;
  formulas.push({ line, displayMode, latex, html });
}
const outsideMath = formulaMarkdown.replace(formulaPattern, '');
if (/(?<!\\)\$/.test(outsideMath) || /\\[()[\]]/.test(outsideMath) || formulas.length === 0) {
  throw new Error('Unpaired math delimiter or unconverted legacy math delimiter');
}
fs.mkdirSync(output, { recursive: true });
const cssDirectory = path.resolve(__dirname, '../web/node_modules/katex/dist');
fs.copyFileSync(path.join(cssDirectory, 'katex.min.css'), path.join(output, 'katex.min.css'));
fs.cpSync(path.join(cssDirectory, 'fonts'), path.join(output, 'fonts'), { recursive: true });
const escapeHtml = value => value.replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
const body = formulas.map(item => `<article><p>Line ${item.line} · ${item.displayMode ? 'display' : 'inline'}</p><div>${item.html}</div><pre>${escapeHtml(item.latex)}</pre></article>`).join('\n');
fs.writeFileSync(path.join(output, 'formula-preview.html'), `<!doctype html><html lang="zh"><meta charset="utf-8"><link rel="stylesheet" href="katex.min.css"><title>初稿 LaTeX 公式核验</title><style>body{max-width:1050px;margin:40px auto;font:16px sans-serif;padding:0 24px}article{padding:18px;border-bottom:1px solid #ddd}pre{white-space:pre-wrap;font-size:13px;color:#666}article>p{color:#555}.katex-display{overflow-x:auto}</style><h1>初稿 LaTeX 公式核验</h1><p>KaTeX ${katex.version} · ${formulas.length} formulas · source Markdown retained</p>${body}</html>`, 'utf8');
const audit = {
  status: 'passed', renderer: `KaTeX ${katex.version}`, source_sha256: crypto.createHash('sha256').update(bytes).digest('hex'),
  inline_formulas: formulas.filter(item => !item.displayMode).length,
  display_formulas: formulas.filter(item => item.displayMode).length,
  delimiter_check: 'passed', syntax_and_rendering: 'passed',
  formulas: formulas.map(({ html, ...item }) => item),
};
fs.writeFileSync(path.join(output, 'formula-audit.json'), JSON.stringify(audit, null, 2), 'utf8');
console.log(JSON.stringify({ status: audit.status, inline: audit.inline_formulas, display: audit.display_formulas }));
