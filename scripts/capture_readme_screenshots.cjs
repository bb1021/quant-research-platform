const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "..");
const outDir = path.join(root, "docs", "screenshots");
fs.mkdirSync(outDir, { recursive: true });

async function waitForApp(page) {
  await page.waitForSelector(".reference-page-label", { timeout: 30000 });
  await page.waitForTimeout(1200);
}

async function gotoSection(page, label) {
  const nav = page.locator('div[role="radiogroup"][aria-label="Section Navigation"]');
  await nav.getByText(label, { exact: true }).click();
  await page.waitForFunction(
    (expected) => document.querySelector(".reference-page-label")?.textContent?.trim() === expected,
    label,
    { timeout: 30000 },
  );
  await page.waitForTimeout(900);
}

async function clickIfVisible(locator) {
  if (!(await locator.count())) {
    return false;
  }
  const first = locator.first();
  if (!(await first.isVisible())) {
    return false;
  }
  await first.click();
  return true;
}

async function waitForAny(page, checks, timeout = 120000) {
  try {
    await page.waitForFunction(
      (needles) => needles.some((needle) => document.body.innerText.includes(needle)),
      checks,
      { timeout },
    );
    return true;
  } catch {
    return false;
  }
}

(async () => {
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 980 }, deviceScaleFactor: 1 });
  page.setDefaultTimeout(30000);

  await page.goto("http://localhost:8501", { waitUntil: "domcontentloaded" });
  await waitForApp(page);

  await page.getByRole("button", { name: "☰" }).click();
  await page.waitForSelector("text=Research Settings", { timeout: 15000 });
  await page.getByRole("button", { name: "Load market data" }).click();
  await waitForAny(page, ["Recent normalised OHLCV records", "Data loaded", "Action failed"], 90000);
  await page.keyboard.press("Escape").catch(() => {});

  await gotoSection(page, "Backtest");
  await clickIfVisible(page.getByRole("button", { name: "Run backtest" }));
  await waitForAny(page, ["Equity Curve", "Backtest complete", "Load market data first"], 45000);

  await gotoSection(page, "AI Research Report");
  await clickIfVisible(page.getByRole("button", { name: "Generate report" }));
  await waitForAny(page, ["Executive Summary", "Report generated", "Load market data first"], 45000);

  const pages = [
    ["Overview", "overview.png"],
    ["Data", "data.png"],
    ["Backtest", "backtest.png"],
    ["Risk Analytics", "risk-analytics.png"],
    ["AI Research Report", "ai-research-report.png"],
  ];

  const captured = [];
  for (const [label, filename] of pages) {
    await gotoSection(page, label);
    await page.waitForTimeout(1400);
    const filePath = path.join(outDir, filename);
    await page.screenshot({ path: filePath, fullPage: false });
    const header = await page.locator(".reference-page-label").innerText();
    const bottom = await page.locator('div[role="radiogroup"][aria-label="Section Navigation"] label:has(input:checked)').innerText();
    captured.push({ filename, header, bottom: bottom.trim(), size: fs.statSync(filePath).size });
  }

  await browser.close();
  console.log(JSON.stringify(captured, null, 2));
})();
