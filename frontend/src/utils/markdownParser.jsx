import React from "react";

/**
 * Render a TeX string using KaTeX. Falls back to raw text if KaTeX isn't loaded.
 * @param {string} tex - Raw LaTeX string
 * @param {boolean} isBlock - Whether to render as display math
 */
export const renderMath = (tex, isBlock) => {
  if (window.katex) {
    try {
      return (
        <span
          dangerouslySetInnerHTML={{
            __html: window.katex.renderToString(tex, {
              displayMode: isBlock,
              throwOnError: false,
            }),
          }}
        />
      );
    } catch (e) {
      console.error(e);
    }
  }
  return isBlock ? (
    <div className="math-block-fallback">{tex}</div>
  ) : (
    <span className="math-inline-fallback">{tex}</span>
  );
};

/**
 * Parse a line of text and render inline elements: math, bold, inline code.
 * Handles both \\( ... \\) and $ ... $ math delimiters.
 */
export const renderInlineElements = (text) => {
  // Split on inline math (\( ... \) or $...$), bold (**...**), or inline code (`...`).
  // Previous pattern was over-escaped and silently matched nothing.
  const inlineParts = text.split(/(\\\([\s\S]*?\\\)|\$[^$\n]+\$|\*\*[^*\n]+\*\*|`[^`\n]+`)/g);
  return inlineParts.map((chunk, index) => {
    if (chunk == null || chunk === "") return null;
    if (chunk.startsWith("\\(") && chunk.endsWith("\\)")) {
      return <React.Fragment key={index}>{renderMath(chunk.slice(2, -2).trim(), false)}</React.Fragment>;
    }
    if (chunk.startsWith("$") && chunk.endsWith("$") && chunk.length > 2) {
      const content = chunk.slice(1, -1).trim();
      // Only treat as math if it contains at least one math-y token, and is not plain currency/prose.
      const mathTokens = /[=+\-*/^_{}\\]|\\frac|\\sqrt|\\int|\\sum|\\alpha|\\beta|\\gamma|\\theta|\\pi|\\lambda|\\mu|\\sigma/;
      const isCurrency = /^\d+(\.\d{1,2})?$/;
      if (isCurrency.test(content) || !mathTokens.test(content)) {
        return chunk;
      }
      return <React.Fragment key={index}>{renderMath(content, false)}</React.Fragment>;
    }
    if (chunk.startsWith("**") && chunk.endsWith("**")) {
      return <strong key={index}>{chunk.slice(2, -2)}</strong>;
    }
    if (chunk.startsWith("`") && chunk.endsWith("`")) {
      return <code key={index}>{chunk.slice(1, -1)}</code>;
    }
    return chunk;
  });
};

/**
 * Parse a text segment and render block-level markdown elements:
 * block math, headings, lists, horizontal rules, paragraphs.
 */
export const parseAndRenderSegment = (segment) => {
  // Split on block math \[ ... \] or $$ ... $$. Previous pattern was over-escaped.
  const parts = segment.split(/(\\\[[\s\S]*?\\\]|\$\$[\s\S]*?\$\$)/g);
  return parts.map((part, index) => {
    if (part.startsWith("\\[") && part.endsWith("\\]")) {
      const tex = part.slice(2, -2).trim();
      return <div key={index} className="math-block">{renderMath(tex, true)}</div>;
    }
    if (part.startsWith("$$") && part.endsWith("$$")) {
      const tex = part.slice(2, -2).trim();
      return <div key={index} className="math-block">{renderMath(tex, true)}</div>;
    }

    const lines = part.split("\n");
    return (
      <React.Fragment key={index}>
        {lines.map((line, j) => {
          const trimmed = line.trim();

          if (trimmed === "") {
            return <div key={j} style={{ height: "6px" }} />;
          }
          if (trimmed === "---") {
            return <hr key={j} className="md-hr" />;
          }
          if (line.startsWith("##### ")) {
            return <h5 key={j} className="md-h5">{renderInlineElements(line.slice(6))}</h5>;
          }
          if (line.startsWith("#### ")) {
            return <h4 key={j} className="md-h4">{renderInlineElements(line.slice(5))}</h4>;
          }
          if (line.startsWith("### ")) {
            return <h3 key={j} className="md-h3">{renderInlineElements(line.slice(4))}</h3>;
          }
          if (line.startsWith("## ")) {
            return <h2 key={j} className="md-h2">{renderInlineElements(line.slice(3))}</h2>;
          }
          if (line.startsWith("# ")) {
            return <h1 key={j} className="md-h1">{renderInlineElements(line.slice(2))}</h1>;
          }

          const listMatch = trimmed.match(/^([\-\*]|\d+\.)\s+(.*)/);
          if (listMatch) {
            const indent = line.length - line.trimStart().length;
            const marker = listMatch[1];
            const content = listMatch[2];
            const isNumbered = /^\d+\.$/.test(marker);
            return (
              <div
                key={j}
                className={`md-list-item ${isNumbered ? "numbered" : "bullet"}`}
                style={{ paddingLeft: `${indent * 8 + 12}px` }}
              >
                {isNumbered ? (
                  <span className="num-prefix">{marker}</span>
                ) : (
                  <span className="bullet-dot">•</span>
                )}
                <span className="bullet-content">{renderInlineElements(content)}</span>
              </div>
            );
          }

          return (
            <p key={j} className="md-p">{renderInlineElements(line)}</p>
          );
        })}
      </React.Fragment>
    );
  });
};

/**
 * Split raw streamed text into typed segments: text, plotly, html, metrics.
 * Each segment carries a type and content for specialized rendering.
 */
export const splitSpecialSegments = (text) => {
  const segments = [];
  let currentPos = 0;

  while (currentPos < text.length) {
    const plotlyIdx = text.indexOf("<!--PLOTLY_JSON-->", currentPos);
    const htmlIdx = text.indexOf("<!--ARTIFACT_HTML-->", currentPos);
    const metricsIdx = text.indexOf("=== PREDICTIVE_METRICS ===", currentPos);

    const candidates = [
      { idx: plotlyIdx, type: "plotly", open: "<!--PLOTLY_JSON-->", close: "<!--/PLOTLY_JSON-->" },
      { idx: htmlIdx, type: "html", open: "<!--ARTIFACT_HTML-->", close: "<!--/ARTIFACT_HTML-->" },
      { idx: metricsIdx, type: "metrics", open: "=== PREDICTIVE_METRICS ===", close: "=== /PREDICTIVE_METRICS ===" },
    ].filter((c) => c.idx !== -1);

    if (candidates.length === 0) {
      segments.push({ type: "text", content: text.substring(currentPos) });
      break;
    }

    candidates.sort((a, b) => a.idx - b.idx);
    const { idx: earliestIdx, type: tagType, open: openTag, close: closeTag } = candidates[0];

    if (earliestIdx > currentPos) {
      segments.push({ type: "text", content: text.substring(currentPos, earliestIdx) });
    }

    const startOfData = earliestIdx + openTag.length;
    const closeIdx = text.indexOf(closeTag, startOfData);

    if (closeIdx !== -1) {
      segments.push({ type: tagType, content: text.substring(startOfData, closeIdx), closed: true });
      currentPos = closeIdx + closeTag.length;
    } else {
      segments.push({ type: tagType, content: text.substring(startOfData), closed: false });
      currentPos = text.length;
    }
  }

  return segments;
};
