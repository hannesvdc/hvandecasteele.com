const MAX_ITEMS = 5;

function parseRSS(xmlText) {
  const xml = new DOMParser().parseFromString(xmlText, "text/xml");
  return [...xml.querySelectorAll("item")].map(item => ({
    title: item.querySelector("title")?.textContent?.trim() ?? "(untitled)",
    link: item.querySelector("link")?.textContent?.trim() ?? "#",
    date: new Date(item.querySelector("pubDate")?.textContent ?? Date.now()),
    description: item.querySelector("description")?.textContent?.trim() ?? ""
  }));
}

async function fetchBlogFeed() {
  const r = await fetch("/blog/feed.xml");
  if (!r.ok) throw new Error(`Blog feed HTTP ${r.status}`);
  const text = await r.text();
  return parseRSS(text).map(x => ({ ...x, source: "blog" }));
}

async function fetchSubstackFeed() {
  const url = "https://api.rss2json.com/v1/api.json?rss_url=" +
    encodeURIComponent("https://hannesvdc.substack.com/feed");

  const r = await fetch(url);
  const j = await r.json();
  return (j.items || []).map(it => ({
    title: it.title,
    link: it.link,
    date: new Date(it.pubDate).toISOString(),
    source: "substack"
  }));
}

// Normalize titles so "Part 1", punctuation, etc. don’t break matching
function normTitle(t) {
  return (t || "")
    .toLowerCase()
    .replace(/&amp;/g, "&")
    .replace(/\bpart\s*i\b/g, "part 1")
    .replace(/\bpart\s*ii\b/g, "part 2")
    .replace(/\bpart\s*iii\b/g, "part 3")
    .replace(/[^\w\s]/g, "")    // remove punctuation
    .replace(/\s+/g, " ")
    .trim();
}

// Merge feeds; prefer blog for main link when both exist
function mergePosts(blogItems, substackItems) {
  const byTitle = new Map();

  // Seed with blog posts (canonical)
  for (const b of blogItems) {
    byTitle.set(normTitle(b.title), {
      title: b.title,
      date: b.date,
      description: b.description || "",
      url: b.link,            // title click target
      blog_url: b.link,
      substack_url: null
    });
  }

  // Add/attach Substack posts
  for (const s of substackItems) {
    const key = normTitle(s.title);

    if (byTitle.has(key)) {
      const x = byTitle.get(key);
      x.substack_url = s.link;

      // Use the more recent date (Substack may be newer than blog or vice versa)
      x.date = (new Date(x.date) > new Date(s.date)) ? x.date : s.date;

      // Keep blog description if present; else allow substack description
      if (!x.description && s.description) x.description = s.description;

      byTitle.set(key, x);
    } else {
      // Substack-only post
      byTitle.set(key, {
        title: s.title,
        date: s.date,
        description: s.description || "",
        url: s.link,           // title click goes to substack
        blog_url: null,
        substack_url: s.link
      });
    }
  }

  return [...byTitle.values()];
}

function render(items) {
  const root = document.getElementById("latest-writing");
  root.innerHTML = "";

  for (const it of items) {
    const article = document.createElement("article");
    article.className = "writing-item";

    // Show the orange Substack button ONLY if there is a blog canonical link
    const showSubstackButton = Boolean(it.blog_url && it.substack_url);

    article.innerHTML = `
      <a class="writing-main" href="${it.url}">
        <h3 class="writing-title">${it.title}</h3>
      </a>
      ${
        showSubstackButton
          ? `<div class="writing-actions">
               <a class="writing-btn" href="${it.substack_url}" target="_blank" rel="noopener">
                 Read on Substack
               </a>
             </div>`
          : ""
      }
    `;

    root.appendChild(article);
  }
}

(async function () {
  let blog = [];
  let sub = [];

  try {
    blog = await fetchBlogFeed();
  } catch (e) {
    console.warn("Blog feed failed:", e);
  }

  try {
    sub = await fetchSubstackFeed();
  } catch (e) {
    console.warn("Substack feed failed:", e);
  }

  const merged = mergePosts(blog, sub);

  if (merged.length === 0) {
    document.getElementById("latest-writing").innerHTML =
      "<div class='writing-item'>No recent writing found.</div>";
    return;
  }

  const newest = merged
    .sort((a, b) => new Date(b.date) - new Date(a.date))
    .slice(0, MAX_ITEMS);

  render(newest);
})();