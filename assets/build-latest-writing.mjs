import fs from "node:fs/promises";

const BLOG_FEED = "https://hvandecasteele.com/blog/feed.xml";
const SUBSTACK_FEED = "https://hannesvdc.substack.com/feed.xml";
const OUTFILE = "/assets/latest.json";
const MAX_ITEMS = 5;

function pick(text, tag) {
  const m = text.match(new RegExp(`<${tag}[^>]*>([\\s\\S]*?)</${tag}>`, "i"));
  return m ? m[1].trim() : "";
}

function stripCdata(s) {
  return s.replace(/^<!\\[CDATA\\[|\\]\\]>$/g, "").trim();
}

function decodeHtml(s) {
  return s
    .replaceAll("&amp;", "&")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&quot;", '"')
    .replaceAll("&#39;", "'");
}

function parseItems(rssText) {
  const items = [...rssText.matchAll(/<item>([\s\S]*?)<\/item>/gi)].map(m => m[1]);

  return items.map(x => {
    const title = decodeHtml(stripCdata(pick(x, "title")));
    const link = stripCdata(pick(x, "link"));
    const pubDate = stripCdata(pick(x, "pubDate"));
    const description = decodeHtml(stripCdata(pick(x, "description"))).replace(/<[^>]+>/g, "").trim();

    return {
      title: title || "(untitled)",
      link: link || "#",
      date: pubDate ? new Date(pubDate).toISOString() : new Date().toISOString(),
      description
    };
  });
}

async function fetchText(url) {
  const r = await fetch(url, { headers: { "User-Agent": "latest-writing-bot" } });
  if (!r.ok) throw new Error(`Failed to fetch ${url}: ${r.status}`);
  return await r.text();
}

function normUrl(u) {
  return (u || "").trim();
}

function normTitle(t) {
  return (t || "")
    .toLowerCase()
    .replace(/[^\w\s]/g, "")
    .replace(/\s+/g, " ")
    .trim();
}

// Heuristic de-dupe: if a blog + substack post have same normalized title, keep blog as canonical
function mergeDedup(blog, substack) {
  const byTitle = new Map();

  for (const b of blog) {
    byTitle.set(normTitle(b.title), { ...b, source: "blog", blog_url: b.link });
  }

  for (const s of substack) {
    const key = normTitle(s.title);
    if (byTitle.has(key)) {
      const existing = byTitle.get(key);
      existing.substack_url = normUrl(s.link);
      byTitle.set(key, existing);
    } else {
      byTitle.set(key, { ...s, source: "substack", substack_url: normUrl(s.link) });
    }
  }

  return [...byTitle.values()];
}

async function main() {
  const [blogRss, subRss] = await Promise.all([fetchText(BLOG_FEED), fetchText(SUBSTACK_FEED)]);
  const blogItems = parseItems(blogRss);
  const subItems = parseItems(subRss);

  const merged = mergeDedup(blogItems, subItems)
    .sort((a, b) => new Date(b.date) - new Date(a.date))
    .slice(0, MAX_ITEMS)
    .map(it => ({
      title: it.title,
      date: it.date,
      description: it.description,
      blog_url: it.blog_url || null,
      substack_url: it.substack_url || (it.source === "substack" ? it.link : null),
      url: it.blog_url || it.substack_url || it.link
    }));

  await fs.mkdir("assets", { recursive: true });
  await fs.writeFile(OUTFILE, JSON.stringify({ updated_at: new Date().toISOString(), items: merged }, null, 2));
  console.log(`Wrote ${OUTFILE} (${merged.length} items)`);
}

main().catch(err => {
  console.error(err);
  process.exit(1);
});