const MAX_ITEMS = 5;

// ---------- Fetchers ----------
async function fetchBlogLatestJson() {
  const r = await fetch("/blog/latest.json", { cache: "no-store" });
  if (!r.ok) throw new Error(`Blog JSON HTTP ${r.status}`);
  const j = await r.json();

  // Normalize to a common schema
  return (j || []).map(p => ({
    title: p.title,
    url: p.url,                         // canonical click target
    date: new Date(p.date),
    substack_url: (p.substack_url || "").trim() || null,
    source: "blog"
  }));
}

async function fetchSubstackLatest() {
  const proxy =
    "https://api.rss2json.com/v1/api.json?rss_url=" +
    encodeURIComponent("https://hannesvdc.substack.com/feed");

  const r = await fetch(proxy, { cache: "no-store" });
  if (!r.ok) throw new Error(`rss2json HTTP ${r.status}`);
  const j = await r.json();

  return (j.items || []).map(it => ({
    title: (it.title || "(untitled)").trim(),
    url: it.link || "#",               // canonical click target (Substack)
    date: new Date(it.pubDate || Date.now()),
    substack_url: it.link || null,     // always exists for Substack posts
    source: "substack"
  }));
}

// ---------- Merge / Dedupe ----------

// For dedupe, we use the explicit blog substack_url mapping as the strongest key.
// That avoids brittle title matching.
function merge(blogItems, subItems) {
  const out = [];

  // 1) Add blog posts first (they define canonical URLs)
  const substackUrlsInBlog = new Set();
  for (const b of blogItems) {
    if (b.substack_url) substackUrlsInBlog.add(b.substack_url);
    out.push(b);
  }

  // 2) Add substack posts that are NOT mirrored by the blog (substack_url matches)
  for (const s of subItems) {
    if (s.substack_url && substackUrlsInBlog.has(s.substack_url)) {
      continue; // skip duplicates of blog posts
    }
    out.push(s);
  }

  return out;
}

// ---------- Render ----------

function render(items) {
  const root = document.getElementById("latest-writing");
  root.innerHTML = "";

  for (const it of items) {
    const article = document.createElement("article");
    article.className = "writing-item";

    const dateStr = it.date instanceof Date && !isNaN(it.date)
      ? it.date.toLocaleDateString("en-US", { month: "short", year: "numeric" })
      : "";

    article.innerHTML = `
      <a class="writing-title" href="${it.url}">
        ${it.title}
      </a>
      ${dateStr ? `<div class="writing-meta">${dateStr}</div>` : ""}

      ${it.substack_url ? `
        <div class="writing-actions">
          <a class="writing-btn" href="${it.substack_url}" target="_blank" rel="noopener">
            Read on Substack
          </a>
        </div>
      ` : ""}
    `;

    root.appendChild(article);
  }
}

// ---------- Main ----------

(async function () {
  let blog = [];
  let sub = [];

  try {
    blog = await fetchBlogLatestJson();
  } catch (e) {
    console.warn("Blog latest.json failed:", e);
  }

  try {
    sub = await fetchSubstackLatest();
  } catch (e) {
    console.warn("Substack feed failed:", e);
  }

  const merged = merge(blog, sub)
    .sort((a, b) => b.date - a.date)
    .slice(0, MAX_ITEMS);

  if (merged.length === 0) {
    document.getElementById("latest-writing").innerHTML =
      "<div class='writing-item'>No recent writing found.</div>";
    return;
  }

  render(merged);
})();