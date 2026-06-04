const fs = require("fs");

const targetUrl = process.argv[2] || "http://localhost:8501";
const screenshotPath = process.argv[3] || "browser_check.png";
const cdpBaseUrl = process.env.CDP_BASE_URL || "http://127.0.0.1:9222";
const scrollY = Number(process.env.SCROLL_Y || "0");
const fullPage = process.env.FULL_PAGE === "1";

async function createPage(url) {
  const endpoint = `${cdpBaseUrl}/json/new?${encodeURIComponent(url)}`;
  const response = await fetch(endpoint, { method: "PUT" });
  if (!response.ok) {
    throw new Error(`Cannot create CDP page: ${response.status} ${await response.text()}`);
  }
  const page = await response.json();
  if (!page.webSocketDebuggerUrl) {
    throw new Error("CDP page did not return a webSocketDebuggerUrl.");
  }
  return page.webSocketDebuggerUrl;
}

async function connect(wsUrl) {
  const ws = new WebSocket(wsUrl);
  await new Promise((resolve, reject) => {
    ws.addEventListener("open", resolve, { once: true });
    ws.addEventListener("error", reject, { once: true });
  });

  let nextId = 0;
  const pending = new Map();

  ws.addEventListener("message", (event) => {
    const message = JSON.parse(event.data);
    if (!message.id || !pending.has(message.id)) {
      return;
    }
    const { resolve, reject } = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) {
      reject(new Error(JSON.stringify(message.error)));
    } else {
      resolve(message.result);
    }
  });

  function send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = ++nextId;
      pending.set(id, { resolve, reject });
      ws.send(JSON.stringify({ id, method, params }));
    });
  }

  return { ws, send };
}

async function main() {
  const wsUrl = await createPage(targetUrl);
  const page = await connect(wsUrl);

  await page.send("Page.enable");
  await page.send("Runtime.enable");
  await page.send("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 1100,
    deviceScaleFactor: 1,
    mobile: false,
  });

  await new Promise((resolve) => setTimeout(resolve, 4000));

  if (scrollY > 0) {
    await page.send("Runtime.evaluate", {
      expression: `window.scrollTo(0, ${scrollY});`,
      returnByValue: true,
    });
    await new Promise((resolve) => setTimeout(resolve, 800));
  }

  const title = await page.send("Runtime.evaluate", {
    expression: "document.title",
    returnByValue: true,
  });
  const text = await page.send("Runtime.evaluate", {
    expression: "document.body.innerText.slice(0, 1200)",
    returnByValue: true,
  });
  const screenshot = await page.send("Page.captureScreenshot", {
    format: "png",
    captureBeyondViewport: fullPage,
  });

  fs.writeFileSync(screenshotPath, Buffer.from(screenshot.data, "base64"));
  page.ws.close();

  console.log(JSON.stringify({
    ok: true,
    url: targetUrl,
    title: title.result.value,
    textPreview: text.result.value,
    screenshotPath,
  }, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
