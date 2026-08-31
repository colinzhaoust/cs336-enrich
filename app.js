(function () {
  const lectures = window.CS336_LECTURES || [];
  const config = window.CS336_SITE_CONFIG || { discussions: { enabled: false } };
  const state = { lecture: 1, filter: "all", query: "", currentSegmentId: "", videoHydrationAfter: 0 };
  if ("scrollRestoration" in history) history.scrollRestoration = "manual";
  const augmentations = new Map();
  const augmentationManifestUrls = [
    "augmentations/lecture_01/manifest.json",
    "augmentations/lecture_02/manifest.json",
    "augmentations/lecture_03/manifest.json"
  ];
  const parseMarkdownSlides = window.CS336_PARSE_MARKDOWN_SLIDES;

  const elements = {
    title: document.querySelector("#lecture-title"),
    kicker: document.querySelector("#lecture-kicker"),
    summary: document.querySelector("#lecture-summary"),
    links: document.querySelector("#lecture-links"),
    outline: document.querySelector("#outline-nav"),
    segments: document.querySelector("#segments"),
    search: document.querySelector("#segment-search"),
    resultCount: document.querySelector("#result-count"),
    empty: document.querySelector("#empty-state"),
    clear: document.querySelector("#clear-filters"),
    readingPosition: document.querySelector("#reading-position"),
    readingChapter: document.querySelector("#reading-chapter"),
    readingSegment: document.querySelector("#reading-segment"),
    readingCount: document.querySelector("#reading-count"),
    readingProgress: document.querySelector("#reading-progress"),
    discussionStatus: document.querySelector("#discussion-status"),
    toast: document.querySelector("#feedback-toast"),
    navToggle: document.querySelector(".nav-toggle"),
    nav: document.querySelector("#course-nav")
  };

  function escapeHTML(value) {
    return String(value)
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#039;");
  }

  function getLecture(number) {
    return lectures.find((lecture) => lecture.number === number) || lectures[0];
  }

  function findSegment(segmentId) {
    for (const lecture of lectures) {
      const segment = lecture.segments.find((item) => item.id === segmentId);
      if (segment) return { lecture, segment };
    }
    return null;
  }

  function segmentNumber(segment, lecture) {
    return lecture.segments.findIndex((item) => item.id === segment.id) + 1;
  }

  function sourceUrlForSegment(segment, lecture) {
    const lineMatch = segment.source.match(/lines?\s+(\d+)(?:[–-](\d+))?/i);
    if (!lineMatch || !/\.py(?:$|\?)/.test(lecture.sourceUrl)) return lecture.sourceUrl;
    const start = lineMatch[1];
    const end = lineMatch[2];
    return `${lecture.sourceUrl}#L${start}${end ? `-L${end}` : ""}`;
  }

  function sourceLinkLabel(segment) {
    if (/lines?\s+\d+/i.test(segment.source)) return "Open cited lines";
    if (/slides?\s+\d+/i.test(segment.source)) return "Open source deck";
    return "Open source material";
  }

  function formatLabel(format) {
    return { manim: "Manim", interactive: "Interactive", slides: "Slides" }[format] || format;
  }

  function jsonSlides(data) {
    const slides = new Map();
    (data.slides || []).forEach((slide) => {
      const details = [
        slide.lede ? `<p>${escapeHTML(slide.lede)}</p>` : "",
        slide.beats?.length ? `<ol>${slide.beats.map((beat) => `<li>${escapeHTML(beat)}</li>`).join("")}</ol>` : "",
        slide.check ? `<p><strong>Check yourself.</strong> ${escapeHTML(slide.check)}</p>` : "",
        slide.code ? `<pre><code>${escapeHTML(slide.code)}</code></pre>` : "",
        slide.equation ? `<p class="slide-equation"><code>${escapeHTML(slide.equation)}</code></p>` : ""
      ].join("");
      slides.set(slide.id, { eyebrow: slide.eyebrow, title: slide.title, html: details });
    });
    return slides;
  }

  function resolveAssetUrl(path, manifestUrl) {
    if (!path) return "";
    const base = path.startsWith("media/") || path.startsWith("augmentations/") || path.startsWith("scenes/")
      ? document.baseURI
      : manifestUrl;
    return new URL(path.split("#")[0], base).href;
  }

  async function fetchJSON(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    return response.json();
  }

  async function fetchText(url) {
    const response = await fetch(url);
    if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
    return response.text();
  }

  async function loadManifest(manifestPath) {
    const manifestUrl = new URL(manifestPath, document.baseURI);
    const manifest = await fetchJSON(manifestUrl);
    let slides = new Map();

    if (manifest.lecture === 1) {
      const slidePath = Object.values(manifest.segments || {})[0]?.slides?.split("#")[0];
      if (slidePath) slides = parseMarkdownSlides(await fetchText(resolveAssetUrl(slidePath, manifestUrl)));
      Object.entries(manifest.segments || {}).forEach(([segmentId, asset]) => {
        augmentations.set(segmentId, {
          scene: asset.scene,
          video: resolveAssetUrl(asset.video, manifestUrl),
          poster: resolveAssetUrl(asset.poster, manifestUrl),
          slide: slides.get(segmentId),
          sharedWith: asset.sharedWith || [],
          caveat: asset.caveat || "",
          source: resolveAssetUrl(asset.source || manifest.render?.source, manifestUrl)
        });
      });
      return;
    }

    if (manifest.lecture === 2) {
      const slidePath = manifest.assets?.[0]?.slides;
      if (slidePath) slides = jsonSlides(await fetchJSON(resolveAssetUrl(slidePath, manifestUrl)));
    } else if (manifest.supplement) {
      slides = parseMarkdownSlides(await fetchText(new URL(manifest.supplement, manifestUrl)));
    }

    (manifest.assets || []).forEach((asset) => {
      const segmentIds = asset.segmentIds || asset.segments || [];
      segmentIds.forEach((segmentId) => {
        augmentations.set(segmentId, {
          scene: asset.scene,
          video: resolveAssetUrl(asset.video, manifestUrl),
          poster: resolveAssetUrl(asset.poster, manifestUrl),
          alt: asset.alt,
          slide: slides.get(segmentId),
          sharedWith: segmentIds.filter((id) => id !== segmentId),
          caveat: asset.caveat || "",
          source: resolveAssetUrl(asset.source, manifestUrl)
        });
      });
    });
  }

  async function loadAugmentations() {
    const results = await Promise.allSettled(augmentationManifestUrls.map(loadManifest));
    const failures = results.filter((result) => result.status === "rejected");
    if (failures.length) console.warn("Some rendered augmentations could not be loaded.", failures.map((result) => result.reason));
  }

  function matches(segment) {
    const text = `${segment.id} ${segment.title} ${segment.summary} ${segment.goal} ${segment.chapter}`.toLowerCase();
    const queryMatch = !state.query || text.includes(state.query.toLowerCase());
    const filterMatch = state.filter === "all"
      || (state.filter === "p0" && segment.priority === "P0")
      || segment.formats.includes(state.filter);
    return queryMatch && filterMatch;
  }

  function visualFloat() {
    return `
      <div class="float-visual" aria-label="Floating point bit allocation comparison">
        <div class="float-row"><strong>FP32</strong><span class="bit sign">1</span><span class="bit exponent wide">8 exponent</span><span class="bit mantissa widest">23 fraction</span></div>
        <div class="float-row"><strong>FP16</strong><span class="bit sign">1</span><span class="bit exponent">5 exponent</span><span class="bit mantissa wide">10 fraction</span></div>
        <div class="float-row"><strong>BF16</strong><span class="bit sign">1</span><span class="bit exponent wide">8 exponent</span><span class="bit mantissa">7 fraction</span></div>
        <div class="number-range"><span>10<sup>−40</sup></span><i></i><b>10<sup>−8</sup></b><i></i><span>10<sup>0</sup></span></div>
      </div>`;
  }

  function visualTensor() {
    return `
      <div class="tensor-visual" aria-label="Named four-dimensional tensor">
        <div class="tensor-cube" aria-hidden="true"><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span><span></span></div>
        <dl>
          <div><dt>B</dt><dd>batch</dd></div><div><dt>S</dt><dd>sequence</dd></div>
          <div><dt>H</dt><dd>heads</dd></div><div><dt>D</dt><dd>head features</dd></div>
        </dl>
      </div>`;
  }

  function visualBPE() {
    return `
      <div class="bpe-visual" aria-label="BPE pair counting and merge preview">
        <div class="token-row"><span>t</span><span class="pair">h</span><span class="pair">e</span><span>_</span><span>c</span><span>a</span><span>t</span><span>_</span><span class="pair">t</span><span class="pair">h</span><span>e</span></div>
        <div class="merge-arrow">most frequent pair <b>→</b> new token 256</div>
        <div class="token-row merged"><span>t</span><span>he</span><span>_</span><span>c</span><span>a</span><span>t</span><span>_</span><span>t</span><span>he</span></div>
      </div>`;
  }

  function visualRoofline() {
    return `
      <div class="roofline-visual">
        <svg viewBox="0 0 640 230" role="img" aria-label="Roofline chart with memory-bound and compute-bound regions">
          <path class="axis" d="M55 20V190H615" />
          <path class="roof" d="M70 178L335 45H605" />
          <path class="knee" d="M335 45V190" />
          <circle cx="112" cy="156" r="7"/><text x="92" y="140">ReLU</text>
          <circle cx="190" cy="118" r="7"/><text x="170" y="102">GELU</text>
          <circle cx="268" cy="78" r="7"/><text x="242" y="64">matvec</text>
          <circle cx="475" cy="45" r="7"/><text x="450" y="31">matmul</text>
          <text class="axis-label" x="255" y="220">arithmetic intensity →</text>
          <text class="region" x="92" y="74">memory-bound</text><text class="region" x="440" y="82">compute-bound</text>
        </svg>
      </div>`;
  }

  function visualMemory() {
    return `
      <div class="memory-visual" aria-label="Per-parameter training memory ledger">
        <span><b>2 B</b> parameter</span><span><b>2 B</b> gradient</span>
        <span><b>4 B</b> first moment</span><span><b>4 B</b> second moment</span>
        <strong>12 bytes / parameter</strong>
      </div>`;
  }

  function visualCheckpoint() {
    return `
      <div class="checkpoint-visual" aria-label="Activation checkpointing sequence">
        <div class="layer-row"><span class="saved">x</span><i></i><span>h1</span><i></i><span>h2</span><i></i><span class="saved">h3</span><i></i><span>h4</span><i></i><span>h5</span><i></i><span class="saved">h6</span></div>
        <div class="recompute-line">backward recomputes only between saved checkpoints</div>
      </div>`;
  }

  function visualRoPE() {
    return `
      <div class="rope-visual" aria-label="Rotary position embedding preview">
        <svg viewBox="0 0 620 230" role="img" aria-label="Two vectors rotated by positional angles">
          <circle cx="165" cy="116" r="80"/><path d="M165 116L223 71"/><path class="second-vector" d="M165 116L121 51"/>
          <path class="arc" d="M197 91A42 42 0 0 0 137 82"/>
          <circle cx="455" cy="116" r="80"/><path d="M455 116L507 61"/><path class="second-vector" d="M455 116L418 45"/>
          <path class="arc" d="M485 84A44 44 0 0 0 431 77"/>
          <text x="92" y="218">position i</text><text x="420" y="218">position j</text>
        </svg>
        <p>attention depends on the relative angle, <strong>i − j</strong></p>
      </div>`;
  }

  function visualGQA() {
    return `
      <div class="gqa-visual" aria-label="Grouped-query attention preview">
        <div class="head-group queries"><b>Q heads</b><span>Q1</span><span>Q2</span><span>Q3</span><span>Q4</span><span>Q5</span><span>Q6</span><span>Q7</span><span>Q8</span></div>
        <div class="sharing-lines"><i></i><i></i><i></i><i></i></div>
        <div class="head-group kv"><b>shared K/V</b><span>KV1</span><span>KV2</span></div>
        <p>fewer K/V groups → smaller cache and less memory traffic</p>
      </div>`;
  }

  function visualWindow() {
    const cells = Array.from({ length: 64 }, (_, index) => {
      const row = Math.floor(index / 8);
      const column = index % 8;
      return `<i class="${Math.abs(row - column) <= 1 ? "visible" : "masked"}"></i>`;
    }).join("");
    return `<div class="window-visual" aria-label="Sliding-window attention matrix"><div class="attention-grid">${cells}</div><p>local band: O(nw) instead of a full n × n pattern</p></div>`;
  }

  function visualFlow(segment) {
    return `<div class="storyboard" aria-label="Storyboard for ${escapeHTML(segment.title)}">${segment.beats.map((beat, index) => `
      <div><span>${String(index + 1).padStart(2, "0")}</span><p>${escapeHTML(beat)}</p></div>`).join("")}</div>`;
  }

  function renderVisual(segment) {
    if (segment.visual === "float") return visualFloat();
    if (segment.visual === "tensor" || segment.visual === "einsum") return visualTensor();
    if (segment.visual === "bpe") return visualBPE();
    if (segment.visual === "roofline") return visualRoofline();
    if (segment.visual === "memory") return visualMemory();
    if (segment.visual === "checkpoint") return visualCheckpoint();
    if (segment.visual === "rope") return visualRoPE();
    if (segment.visual === "gqa") return visualGQA();
    if (segment.visual === "window") return visualWindow();
    return visualFlow(segment);
  }

  function sourceContent(segment, lecture) {
    const sourceUrl = sourceUrlForSegment(segment, lecture);
    return `
      <p class="source-location"><span>Source locator</span>${escapeHTML(segment.source)}</p>
      <p>${escapeHTML(segment.summary)}</p>
      ${segment.code ? `<pre><code>${escapeHTML(segment.code)}</code></pre>` : ""}
      <a class="text-link" href="${escapeHTML(sourceUrl)}" target="_blank" rel="noreferrer" aria-label="${escapeHTML(sourceLinkLabel(segment))} for ${escapeHTML(segment.title)} in a new tab">${sourceLinkLabel(segment)} <span aria-hidden="true">↗</span></a>`;
  }

  function sequenceMap(segment) {
    return `
      <div class="sequence-map" aria-label="Visual explanation sequence">
        <p><span>From the source claim</span> The visual proceeds through three inspectable steps.</p>
        <ol>${segment.beats.map((beat, index) => `<li><span>${index + 1}</span>${escapeHTML(beat)}</li>`).join("")}</ol>
      </div>`;
  }

  function sharedSequence(segment, asset, lecture) {
    if (!asset.sharedWith.length) return "";
    const ids = [segment.id, ...asset.sharedWith]
      .filter((id, index, items) => items.indexOf(id) === index)
      .sort((a, b) => lecture.segments.findIndex((item) => item.id === a) - lecture.segments.findIndex((item) => item.id === b));
    return `
      <nav class="shared-sequence" aria-label="Source segments sharing this animation">
        <div><strong>One clip, ${ids.length} source segments</strong><span>The clip is shared; each row keeps its own source claim and slide notes.</span></div>
        <ol>${ids.map((id) => {
          const found = findSegment(id);
          const current = id === segment.id;
          return `<li><a href="#${escapeHTML(id)}"${current ? ' aria-current="step"' : ""}><span>${escapeHTML(id)}</span>${escapeHTML(found?.segment.title || id)}</a></li>`;
        }).join("")}</ol>
      </nav>`;
  }

  function renderedAugmentation(segment, asset, lecture) {
    const descriptionId = `${segment.id}-media-description`;
    const slide = asset.slide ? `
      <details class="slide-supplement">
        <summary aria-expanded="false"><span>Segment-specific notes</span><strong>${escapeHTML(asset.slide.title || segment.title)}</strong><small>Toggle</small></summary>
        <div class="slide-copy">
          ${asset.slide.eyebrow ? `<p class="micro-label">${escapeHTML(asset.slide.eyebrow)}</p>` : ""}
          ${asset.slide.html}
        </div>
      </details>` : "";

    return `
      ${sequenceMap(segment)}
      ${sharedSequence(segment, asset, lecture)}
      <figure class="augmentation-media" data-augmentation="rendered">
        <div class="media-label"><span>Local Manim render</span><span>Silent · use controls to pause or scrub</span></div>
        <video controls playsinline preload="none" poster="${escapeHTML(asset.poster)}" aria-describedby="${escapeHTML(descriptionId)}" data-lazy-video>
          <source data-src="${escapeHTML(asset.video)}" type="video/mp4">
          Your browser cannot play this MP4. <a href="${escapeHTML(asset.video)}">Open the animation directly</a>.
        </video>
        <figcaption id="${escapeHTML(descriptionId)}">
          <span><strong>${escapeHTML(asset.scene)}</strong> · rendered from repository Manim code</span>
          <span class="media-links"><a href="${escapeHTML(asset.video)}">Open MP4</a>${asset.source ? `<a href="${escapeHTML(asset.source)}">Manim source</a>` : ""}</span>
        </figcaption>
        ${asset.alt ? `<p class="media-description">${escapeHTML(asset.alt)}</p>` : ""}
        ${asset.caveat ? `<p class="media-caveat"><strong>Boundary.</strong> ${escapeHTML(asset.caveat)}</p>` : ""}
      </figure>
      ${slide}`;
  }

  function segmentTemplate(segment, lecture) {
    const feedbackLabel = config.discussions.enabled ? "Discuss this segment" : "Copy feedback ID";
    const augmentation = augmentations.get(segment.id);
    const position = segmentNumber(segment, lecture);
    const previous = lecture.segments[position - 2];
    const next = lecture.segments[position];
    return `
      <article id="${escapeHTML(segment.id)}" class="segment" data-chapter="${escapeHTML(segment.chapter)}">
        <div class="segment-meta">
          <p class="segment-position"><span>${String(position).padStart(2, "0")}</span> / ${String(lecture.segments.length).padStart(2, "0")}</p>
          <a class="segment-anchor" href="#${escapeHTML(segment.id)}" aria-label="Link to ${escapeHTML(segment.title)}">${escapeHTML(segment.id)}</a>
          <p class="segment-chapter">${escapeHTML(segment.chapter)}</p>
          <div class="tag-row"><span class="priority ${segment.priority.toLowerCase()}">${segment.priority}</span>${segment.formats.map((format) => `<span>${formatLabel(format)}</span>`).join("")}</div>
          <button type="button" class="copy-link-button" data-copy-link="${escapeHTML(segment.id)}">Copy stable link</button>
          <button type="button" class="discuss-button" data-discuss="${escapeHTML(segment.id)}">${feedbackLabel}</button>
          <nav class="segment-pager" aria-label="Adjacent lecture segments">
            ${previous ? `<a href="#${escapeHTML(previous.id)}" aria-label="Previous segment: ${escapeHTML(previous.title)}">← Previous</a>` : `<span></span>`}
            ${next ? `<a href="#${escapeHTML(next.id)}" aria-label="Next segment: ${escapeHTML(next.title)}">Next →</a>` : `<span></span>`}
          </nav>
        </div>
        <section class="source-panel" aria-labelledby="${escapeHTML(segment.id)}-source">
          <p class="panel-step"><span>1</span> Read the source claim</p>
          <h3 id="${escapeHTML(segment.id)}-source">${escapeHTML(segment.title)}</h3>
          ${sourceContent(segment, lecture)}
        </section>
        <section class="augmented-panel" aria-labelledby="${escapeHTML(segment.id)}-augmented">
          <p class="panel-step"><span>2</span> Inspect the augmentation</p>
          <div class="augmented-heading">
            <div><p class="micro-label">Learning goal</p><h3 id="${escapeHTML(segment.id)}-augmented">${escapeHTML(segment.goal)}</h3></div>
            <span class="status${augmentation ? " rendered" : ""}">${augmentation ? "Rendered clip" : "Planned sequence"}</span>
          </div>
          ${augmentation ? renderedAugmentation(segment, augmentation, lecture) : renderVisual(segment)}
          <div class="discussion-slot" data-discussion-slot="${escapeHTML(segment.id)}" hidden></div>
        </section>
      </article>`;
  }

  function renderOutline(lecture) {
    elements.outline.innerHTML = lecture.chapters.map((chapter, index) => {
      const count = lecture.segments.filter((segment) => segment.chapter === chapter).length;
      return `<button type="button" data-chapter-target="${escapeHTML(chapter)}"><span>${index + 1} · ${String(count).padStart(2, "0")}</span>${escapeHTML(chapter)}</button>`;
    }).join("");
  }

  function loadVideo(video) {
    const source = video.querySelector("source[data-src]");
    if (!source || source.hasAttribute("src")) return;
    source.src = source.dataset.src;
    video.load();
  }

  let videoFrame = 0;
  let videoDelayTimer = 0;

  function hydrateNearbyVideos() {
    const delay = state.videoHydrationAfter - performance.now();
    if (delay > 0) {
      window.clearTimeout(videoDelayTimer);
      videoDelayTimer = window.setTimeout(hydrateNearbyVideos, delay + 20);
      return;
    }
    window.cancelAnimationFrame(videoFrame);
    videoFrame = window.requestAnimationFrame(() => {
      document.querySelectorAll("video[data-lazy-video]").forEach((video) => {
        const bounds = video.getBoundingClientRect();
        if (bounds.top <= window.innerHeight + 600 && bounds.bottom >= -600) loadVideo(video);
      });
    });
  }

  function prepareVideos() {
    hydrateNearbyVideos();
  }

  function updateReadingPosition(segmentId) {
    const lecture = getLecture(state.lecture);
    const segment = lecture.segments.find((item) => item.id === segmentId);
    if (!segment) return;
    const position = segmentNumber(segment, lecture);
    state.currentSegmentId = segment.id;
    elements.readingChapter.textContent = `Lecture ${lecture.number} · ${segment.chapter}`;
    elements.readingSegment.textContent = segment.title;
    elements.readingCount.textContent = `${position} / ${lecture.segments.length}`;
    elements.readingProgress.max = lecture.segments.length;
    elements.readingProgress.value = position;
    elements.readingProgress.setAttribute("aria-valuetext", `Segment ${position} of ${lecture.segments.length}: ${segment.title}`);
    document.querySelectorAll("[data-chapter-target]").forEach((button) => {
      const active = button.dataset.chapterTarget === segment.chapter;
      button.classList.toggle("is-current", active);
      if (active) button.setAttribute("aria-current", "location");
      else button.removeAttribute("aria-current");
    });
    document.querySelectorAll(".segment").forEach((article) => article.classList.toggle("is-current", article.id === segment.id));
  }

  function revealSegment(segmentId) {
    const reveal = () => {
      const target = document.getElementById(segmentId);
      if (!target) return;
      target.scrollIntoView({ block: "start", behavior: "auto" });
      updateReadingPosition(segmentId);
    };
    window.requestAnimationFrame(() => window.requestAnimationFrame(reveal));
    document.fonts?.ready.then(() => window.requestAnimationFrame(reveal));
    window.setTimeout(reveal, 420);
  }

  let readingFrame = 0;

  function syncReadingPosition() {
    window.cancelAnimationFrame(readingFrame);
    readingFrame = window.requestAnimationFrame(() => {
      const articles = [...document.querySelectorAll(".segment")];
      if (!articles.length) return;
      const marker = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--header-height")) * 16 + 112;
      const current = articles.find((article) => article.getBoundingClientRect().bottom > marker) || articles.at(-1);
      updateReadingPosition(current.id);
    });
  }

  function renderSegments(lecture) {
    const visible = lecture.segments.filter(matches);
    elements.segments.innerHTML = visible.map((segment) => segmentTemplate(segment, lecture)).join("");
    elements.empty.hidden = visible.length > 0;
    const renderedSegments = visible.filter((segment) => augmentations.has(segment.id));
    const renderedClips = new Set(renderedSegments.map((segment) => augmentations.get(segment.id).video)).size;
    elements.resultCount.textContent = `${visible.length} of ${lecture.segments.length} segments · ${renderedClips} rendered clip${renderedClips === 1 ? "" : "s"} cover ${renderedSegments.length} segment${renderedSegments.length === 1 ? "" : "s"}`;
    prepareVideos();
    const hashSegment = findSegment(window.location.hash.slice(1));
    const initial = hashSegment?.lecture.number === lecture.number && visible.some((segment) => segment.id === hashSegment.segment.id)
      ? hashSegment.segment.id
      : visible[0]?.id;
    if (initial) updateReadingPosition(initial);
  }

  function setNavState() {
    document.querySelectorAll("[data-lecture]").forEach((button) => {
      const active = Number(button.dataset.lecture) === state.lecture;
      button.classList.toggle("is-active", active);
      if (active) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    });
  }

  function renderLecture(options = {}) {
    const lecture = getLecture(state.lecture);
    state.currentSegmentId = "";
    elements.kicker.textContent = `Lecture ${lecture.number} · ${lecture.instructor}`;
    elements.title.textContent = lecture.title;
    elements.summary.textContent = lecture.summary;
    elements.links.innerHTML = `
      <a href="${escapeHTML(lecture.sourceUrl)}" target="_blank" rel="noreferrer">Original material ↗</a>
      <a href="${escapeHTML(lecture.videoUrl)}" target="_blank" rel="noreferrer">Official video ↗</a>`;
    renderOutline(lecture);
    renderSegments(lecture);
    setNavState();
    document.title = `Lecture ${lecture.number}: ${lecture.title} · CS336 Enriched`;
    if (!options.preserveScroll) window.scrollTo({ top: 0, behavior: "auto" });
  }

  function showToast(message) {
    elements.toast.textContent = message;
    elements.toast.hidden = false;
    window.clearTimeout(showToast.timer);
    showToast.timer = window.setTimeout(() => { elements.toast.hidden = true; }, 3200);
  }

  async function copyTemplate(segment) {
    const template = `Segment: ${segment.id}\nType: confusion | correction | visual idea\nI expected:\nI got lost when:\nA clearer visual might:`;
    try {
      await navigator.clipboard.writeText(template);
      showToast(`${segment.id} feedback template copied`);
    } catch (_error) {
      showToast(`Use segment ID ${segment.id} in your comment`);
    }
  }

  async function copyStableLink(segmentId) {
    const found = findSegment(segmentId);
    if (!found) return;
    const url = new URL(window.location.href);
    url.hash = segmentId;
    try {
      await navigator.clipboard.writeText(url.href);
      showToast(`${segmentId} stable link copied`);
    } catch (_error) {
      showToast(`Stable link: ${url.href}`);
    }
  }

  function mountGiscus(slot, segment) {
    slot.replaceChildren();
    const script = document.createElement("script");
    script.src = "https://giscus.app/client.js";
    script.dataset.repo = config.discussions.repo;
    script.dataset.repoId = config.discussions.repoId;
    script.dataset.category = config.discussions.category;
    script.dataset.categoryId = config.discussions.categoryId;
    script.dataset.mapping = "specific";
    script.dataset.term = segment.id;
    script.dataset.strict = "1";
    script.dataset.reactionsEnabled = "1";
    script.dataset.emitMetadata = "0";
    script.dataset.inputPosition = "top";
    script.dataset.theme = "light";
    script.dataset.lang = "en";
    script.dataset.loading = "lazy";
    script.crossOrigin = "anonymous";
    script.async = true;
    slot.append(script);
  }

  async function handleDiscussion(segmentId) {
    const found = findSegment(segmentId);
    if (!found) return;
    await copyTemplate(found.segment);
    const slot = document.querySelector(`[data-discussion-slot="${segmentId}"]`);
    if (!slot) return;
    document.querySelectorAll(".discussion-slot").forEach((item) => {
      if (item !== slot) item.hidden = true;
    });
    slot.hidden = false;
    if (config.discussions.enabled && config.discussions.repoId && config.discussions.categoryId) {
      mountGiscus(slot, found.segment);
    } else {
      slot.innerHTML = `<div class="discussion-pending"><strong>Discussion linking is prepared.</strong><p>The feedback template is copied. Once the public repository and Giscus IDs are configured, this area becomes the live discussion for <code>${escapeHTML(segmentId)}</code>.</p><a href="#feedback-guide">Read the feedback workflow</a></div>`;
    }
    slot.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  document.addEventListener("click", (event) => {
    const lectureButton = event.target.closest("[data-lecture]");
    if (lectureButton) {
      state.lecture = Number(lectureButton.dataset.lecture);
      state.query = "";
      state.filter = "all";
      elements.search.value = "";
      document.querySelectorAll(".filter-button").forEach((button) => button.classList.toggle("is-active", button.dataset.filter === "all"));
      history.replaceState(null, "", `#lecture-${state.lecture}`);
      renderLecture();
      elements.navToggle.setAttribute("aria-expanded", "false");
      elements.nav.classList.remove("is-open");
      return;
    }

    const filterButton = event.target.closest("[data-filter]");
    if (filterButton) {
      state.filter = filterButton.dataset.filter;
      document.querySelectorAll(".filter-button").forEach((button) => button.classList.toggle("is-active", button === filterButton));
      renderSegments(getLecture(state.lecture));
      return;
    }

    const chapterButton = event.target.closest("[data-chapter-target]");
    if (chapterButton) {
      const target = document.querySelector(`[data-chapter="${CSS.escape(chapterButton.dataset.chapterTarget)}"]`);
      target?.scrollIntoView({ behavior: "smooth", block: "start" });
      return;
    }

    const discussButton = event.target.closest("[data-discuss]");
    if (discussButton) {
      handleDiscussion(discussButton.dataset.discuss);
      return;
    }

    const copyLinkButton = event.target.closest("[data-copy-link]");
    if (copyLinkButton) copyStableLink(copyLinkButton.dataset.copyLink);
  });

  elements.search.addEventListener("input", (event) => {
    state.query = event.target.value.trim();
    renderSegments(getLecture(state.lecture));
  });

  elements.clear.addEventListener("click", () => {
    state.query = "";
    state.filter = "all";
    elements.search.value = "";
    document.querySelectorAll(".filter-button").forEach((button) => button.classList.toggle("is-active", button.dataset.filter === "all"));
    renderSegments(getLecture(state.lecture));
    elements.search.focus();
  });

  elements.navToggle.addEventListener("click", () => {
    const open = elements.navToggle.getAttribute("aria-expanded") === "true";
    elements.navToggle.setAttribute("aria-expanded", String(!open));
    elements.nav.classList.toggle("is-open", !open);
  });

  document.addEventListener("keydown", (event) => {
    const summary = event.target.closest?.(".slide-supplement summary");
    if (summary && (event.key === "Enter" || event.key === " ")) {
      event.preventDefault();
      const details = summary.closest("details");
      details.open = !details.open;
      summary.setAttribute("aria-expanded", String(details.open));
      return;
    }
    if (event.key !== "Escape" || elements.navToggle.getAttribute("aria-expanded") !== "true") return;
    elements.navToggle.setAttribute("aria-expanded", "false");
    elements.nav.classList.remove("is-open");
    elements.navToggle.focus();
  });

  document.addEventListener("toggle", (event) => {
    if (!event.target.matches?.(".slide-supplement")) return;
    event.target.querySelector("summary")?.setAttribute("aria-expanded", String(event.target.open));
  }, true);

  window.addEventListener("scroll", () => {
    syncReadingPosition();
    hydrateNearbyVideos();
  }, { passive: true });
  window.addEventListener("resize", () => {
    syncReadingPosition();
    hydrateNearbyVideos();
  });

  window.addEventListener("hashchange", () => {
    const hash = window.location.hash.slice(1);
    const segmentMatch = findSegment(hash);
    const lectureMatch = hash.match(/^lecture-(\d)$/);

    if (segmentMatch) {
      state.videoHydrationAfter = performance.now() + 800;
      const mustRender = state.lecture !== segmentMatch.lecture.number || state.query || state.filter !== "all" || !document.getElementById(hash);
      if (mustRender) {
        state.lecture = segmentMatch.lecture.number;
        state.query = "";
        state.filter = "all";
        elements.search.value = "";
        document.querySelectorAll(".filter-button").forEach((button) => button.classList.toggle("is-active", button.dataset.filter === "all"));
        renderLecture({ preserveScroll: true });
      }
      window.setTimeout(() => {
        revealSegment(hash);
      }, 120);
      return;
    }

    if (lectureMatch && Number(lectureMatch[1]) !== state.lecture) {
      state.lecture = Number(lectureMatch[1]);
      state.query = "";
      state.filter = "all";
      elements.search.value = "";
      renderLecture();
    }
  });

  async function initialize() {
    const hash = window.location.hash.slice(1);
    const segmentMatch = findSegment(hash);
    const lectureMatch = hash.match(/^lecture-(\d)$/);
    if (segmentMatch) state.lecture = segmentMatch.lecture.number;
    else if (lectureMatch) state.lecture = Number(lectureMatch[1]);
    if (segmentMatch) state.videoHydrationAfter = performance.now() + 800;

    const configured = Boolean(config.discussions.enabled && config.discussions.repoId && config.discussions.categoryId);
    elements.discussionStatus.innerHTML = configured
      ? `<strong>Live:</strong> comments are stored in GitHub Discussions, one discussion per segment.`
      : `<strong>Setup pending:</strong> create the public repository, enable Discussions, install Giscus, then add the repository and category IDs in <code>config.js</code>.`;

    renderLecture({ preserveScroll: Boolean(segmentMatch) });
    if (segmentMatch) window.setTimeout(() => {
      revealSegment(hash);
    }, 50);
    await loadAugmentations();
    renderSegments(getLecture(state.lecture));
    if (segmentMatch) window.setTimeout(() => {
      revealSegment(hash);
    }, 160);
  }

  initialize();
})();
