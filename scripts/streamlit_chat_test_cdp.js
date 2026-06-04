const cdpBaseUrl = process.env.CDP_BASE_URL || "http://127.0.0.1:9222";
const targetUrl = process.argv[2] || "http://localhost:8501/哀伤支持对话";
const testMessage = process.argv[3] || "我今天还是很难过，什么都不想说。";
const expectedText = process.argv[4] || "本轮未调用具体回忆";

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

async function waitFor(page, expression, timeoutMs = 120000) {
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

  const inputBox = await page.send("Runtime.evaluate", {
    expression: `
      (() => {
        const textareas = Array.from(document.querySelectorAll('textarea'));
        const target = textareas.find(el => (el.getAttribute('aria-label') || '').includes('你想对它说什么')) || textareas[textareas.length - 1];
        if (!target) return { ok: false, count: textareas.length };
        target.scrollIntoView({ block: 'center' });
        target.focus();
        const setter = Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set;
        setter.call(target, '');
        target.dispatchEvent(new InputEvent('input', { bubbles: true, inputType: 'deleteContentBackward', data: null }));
        const rect = target.getBoundingClientRect();
        return { ok: true, x: rect.left + 12, y: rect.top + 12 };
      })()
    `,
    returnByValue: true,
  });
  if (!inputBox.result.value?.ok) throw new Error(`Cannot find chat textarea: ${JSON.stringify(inputBox.result.value)}`);

  await page.send("Input.dispatchMouseEvent", { type: "mousePressed", x: inputBox.result.value.x, y: inputBox.result.value.y, button: "left", clickCount: 1 });
  await page.send("Input.dispatchMouseEvent", { type: "mouseReleased", x: inputBox.result.value.x, y: inputBox.result.value.y, button: "left", clickCount: 1 });
  await page.send("Input.insertText", { text: testMessage });
  await new Promise((resolve) => setTimeout(resolve, 500));

  const sendButton = await page.send("Runtime.evaluate", {
    expression: `
      (() => {
        const send = Array.from(document.querySelectorAll('button')).find(button => button.innerText.trim() === '发送');
        if (!send) return { ok: false };
        const rect = send.getBoundingClientRect();
        return { ok: true, x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 };
      })()
    `,
    returnByValue: true,
  });
  if (!sendButton.result.value?.ok) throw new Error("Cannot locate send button.");

  await page.send("Input.dispatchMouseEvent", { type: "mousePressed", x: sendButton.result.value.x, y: sendButton.result.value.y, button: "left", clickCount: 1 });
  await page.send("Input.dispatchMouseEvent", { type: "mouseReleased", x: sendButton.result.value.x, y: sendButton.result.value.y, button: "left", clickCount: 1 });

  await waitFor(page, `document.body.innerText.includes(${JSON.stringify(testMessage)})`);
  await waitFor(page, `document.body.innerText.includes(${JSON.stringify(expectedText)})`);

  const textResult = await page.send("Runtime.evaluate", {
    expression: "document.body.innerText",
    returnByValue: true,
  });
  const text = textResult.result.value || "";
  page.ws.close();
  console.log(JSON.stringify({
    ok: true,
    sawExpectedText: text.includes(expectedText),
    sawUserMessage: text.includes(testMessage),
    textPreview: text.slice(0, 1800),
  }, null, 2));
}

main().catch((error) => {
  console.error(error.stack || error.message);
  process.exit(1);
});
