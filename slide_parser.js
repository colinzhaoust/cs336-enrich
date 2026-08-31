(function () {
  function escapeHTML(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function inlineMarkdown(value) {
    return escapeHTML(value)
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  }

  function markdownBlocks(lines) {
    const blocks = [];
    let index = 0;

    while (index < lines.length) {
      const line = lines[index].trim();
      if (!line) {
        index += 1;
        continue;
      }

      if (line.startsWith("|") && lines[index + 1]?.trim().match(/^\|?[\s:|-]+\|$/)) {
        const rows = [];
        while (index < lines.length && lines[index].trim().startsWith("|")) {
          rows.push(lines[index].trim().replace(/^\||\|$/g, "").split("|").map((cell) => cell.trim()));
          index += 1;
        }
        const header = rows[0] || [];
        const body = rows.slice(2);
        blocks.push(`<div class="slide-table-wrap"><table><thead><tr>${header.map((cell) => `<th>${inlineMarkdown(cell)}</th>`).join("")}</tr></thead><tbody>${body.map((row) => `<tr>${row.map((cell) => `<td>${inlineMarkdown(cell)}</td>`).join("")}</tr>`).join("")}</tbody></table></div>`);
        continue;
      }

      const paragraph = [];
      while (index < lines.length && lines[index].trim() && !lines[index].trim().startsWith("|")) {
        paragraph.push(lines[index].trim());
        index += 1;
      }
      blocks.push(`<p>${inlineMarkdown(paragraph.join(" "))}</p>`);
    }

    return blocks.join("");
  }

  window.CS336_PARSE_MARKDOWN_SLIDES = function parseMarkdownSlides(markdown) {
    const slides = new Map();
    const lines = markdown.split(/\r?\n/);
    let heading = "";
    let body = [];

    function commit() {
      if (!heading) return;
      const ids = heading.match(/L\d{2}-[A-Z0-9-]+/g) || [];
      const label = heading
        .replace(/L\d{2}-[A-Z0-9-]+/g, "")
        .replace(/[\/·]+/g, " ")
        .trim();
      const titlePattern = /^\*\*Slide title:\*\*\s*(.+)$/i;
      const explicitTitle = body.map((line) => line.trim().match(titlePattern)?.[1]?.trim()).find(Boolean);
      const content = markdownBlocks(body.filter((line) => !titlePattern.test(line.trim())));
      ids.forEach((id) => slides.set(id, { title: explicitTitle || label || id, html: content }));
    }

    lines.forEach((line) => {
      if (line.startsWith("## ")) {
        commit();
        heading = line.slice(3).trim();
        body = [];
      } else if (heading) {
        body.push(line);
      }
    });
    commit();
    return slides;
  };
})();
