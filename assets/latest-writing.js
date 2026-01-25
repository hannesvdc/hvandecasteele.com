const BLOG_FEED = "/blog/feed.xml";
const SUBSTACK_FEED = "https://substack.com/@hannesvdc/feed.xml";
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

async function fetchFeed(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`Failed to fetch ${url}`);
  return parseRSS(await r.text());
}

function render(items) {
  const root = document.getElementById("latest-writing");
  root.innerHTML = "";

  for (const it of items) {
    const article = document.createElement("article");
    article.className = "writing-item";

    article.innerHTML = `
      <a class="writing-main" href="${it.link}">
        <h3 class="writing-title">${it.title}</h3>
        ${it.description ? `<p class="writing-subtitle">${it.description}</p>` : ""}
      </a>
      ${
        it.source === "substack"
          ? `<div class="writing-actions">
               <a class="writing-btn" href="${it.link}" target="_blank" rel="noopener">
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
  let items = [];

  try {
    const blog = await fetchFeed(BLOG_FEED);
    items.push(...blog.map(x => ({ ...x, source: "blog" })));
  } catch (e) {
    console.warn("Blog feed failed:", e);
  }

  try {
    const sub = await fetchFeed(SUBSTACK_FEED);
    items.push(...sub.map(x => ({ ...x, source: "substack" })));
  } catch (e) {
    console.warn("Substack feed failed (likely CORS):", e);
  }

  if (items.length === 0) {
    document.getElementById("latest-writing").innerHTML =
      "<div class='writing-item'>No recent writing found.</div>";
    return;
  }

  // Sort by date and take newest
  items
    .sort((a, b) => b.date - a.date)
    .slice(0, MAX_ITEMS);

  render(items.slice(0, MAX_ITEMS));
})();