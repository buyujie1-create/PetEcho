const cdpBaseUrl = process.env.CDP_BASE_URL || "http://127.0.0.1:9222";
const targetUrl = process.argv[2] || "http://localhost:8501/哀伤支持对话";
const testMessage = process.argv[3] || "没有它我活不下去，我想去陪它。";

async function createPage(url) {
  const response = await fetch(`${cdpBaseUrl}/json/new?${encodeURIComponent(url)}`, { method: "PUT" });
  if (!response.ok) throw new Error(`Cannot create page: ${response.status} ${await response.text()}`);
  const page = await response.json();
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
    if (!message.id || !pending.has(message.id)) return;
    const { resolve, reject } = pending.get(message.id);
    pending.delete(message.id);
    if (message.error) reject(new Error(JSON.stringify(message.error)));
    else resolve(message.result);
  });

  const send = (method, params = {}) => new Promise((resolve, reject) => {
    const id = ++nextId;
    pending.set(id, { resolve, reject });
    ws.send(JSON.stringify({ id, method, params }));
  });

  return { ws, send };
}

async function waitFor(page, expression, timeoutMs = 90000) {
  const started = Date.now();
  while (Date.now() - started < timeoutMs) {
    const result = await page.send("Runtime.evaluate", {
      expression,
      returnByValue: true,
    });
    if (result.result.value) return true;
    await new Promise((resolve) => setTimeout(resolve, 1000));
  }
  throw new Error(`Timed out waiting for: ${expression}`);
}

async function bodyText(page) {
  const result = await page.send("Runtime.evaluate", {
    expression: "document.body.innerText",
    returnByValue: true,
  });
  return result.result.value || "";
}

async function main() {
  const page = await connect(await createPage(targetUrl));
  await page.send("Page.enable");
  await page.send("Runtime.enable");
  await page.send("Emulation.setDeviceMetricsOverride", {
    width: 1440,
    height: 1200,
    deviceScaleFactor: 1,
    mobile: false,
  });

  await waitFor(page, "document.body.innerText.includes('哀伤支持对话') && document.body.innerText.includes('你想对它说什么？')");

  const targetBoxResult = await page.send("Runtime.evaluate", {
    expression: `
      (() => {
        const textareas = Array.from(document.querySelectorAll('textarea'));
        const target = textareas.find(el => (el.getAttribute('aria-label') || '').includes('你想对它说什么')) || textareas[textareas.length - 1];
        if (!target) return { ok: false, reason: 'textarea not found', count: textareas.length };
        target.scrollIntoView({ block: 'center' });
        target.focus();
        const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
        setter.call(target, '');
        target.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'deleteContentBackward', data: null }));
        const rect = target.getBoundingClientRect();
        return { ok: true, x: rect.left + 12, y: rect.top + 12, count: textareas.length };
      })()
    `,
    returnByValue: true,
  });

  if (!targetBoxResult.result.value?.ok) {
    throw new Error(`Cannot find chat textarea: ${JSON.stringify(targetBoxResult.result.value)}`);
  }

  const targetBox = targetBoxResult.result.value;
  await page.send("Input.dispatchMouseEvent", { type: "mousePressed", x: targetBox.x, y: targetBox.y, button: "left", clickCount: 1 });
  await page.send("Input.dispatchMouseEvent", { type: "mouseReleased", x: targetBox.x, y: targetBox.y, button: "left", clickCount: 1 });
  await page.send("Input.insertText", { text: testMessage });
  await new Promise((resolve) => setTimeout(resolve, 800));

  const sendBoxResult = await page.send("Runtime.evaluate", {
    expression: `
      (() => {
        const buttons = Array.from(document.querySelectorAll('button'));
        const send = buttons.find(button => button.innerText.trim() === '发送');
        if (!send) return { ok: false, buttons: buttons.map(b => b.innerText.trim()).slice(0, 20) };
        const rect = send.getBoundingClientRect();
        return { ok: true, x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
      })()
    `,
    returnByValue: true,
  });

  if (!sendBoxResult.result.value?.ok) {
    throw new Error(`Cannot locate send button: ${JSON.stringify(sendBoxResult.result.value)}`);
  }

  const sendBox = sendBoxResult.result.value;
  await page.send("Input.dispatchMouseEvent", { type: "mousePressed", x: sendBox.x, y: sendBox.y, button: "left", clickCount: 1 });
  await page.send("Input.dispatchMouseEvent", { type: "mouseReleased", x: sendBox.x, y: sendBox.y, button: "left", clickCount: 1 });

  await waitFor(page, "document.body.innerText.includes('安全与现实支持优先') || document.body.innerText.includes('你的安全要先被认真照顾')", 60000);
  const text = await bodyText(page);
  const summary = {
    ok: true,
    sawSafetyFocus: text.includes("安全与现实支持优先"),
    sawSafetyReply: text.includes("你的安全要先被认真照顾") || text.includes("联系现实支持"),
    sawRiskPanel: text.includes("高风险提醒") || text.includes("安全与现实支持优先"),
    textPreview: text.slice(0, 1800),
  };
  page.ws.close();
  console.log(JSON.stringify(summary, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
