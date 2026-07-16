const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const root = path.resolve(__dirname, "..");
const outDir = path.join(root, "docs", "screenshots");
fs.mkdirSync(outDir, { recursive: true });

async function waitForReady(page) {
  await page.waitForSelector("text=Quant Research Platform", { timeout: 30000 });
  await page.waitForTimeout(1500);
}

async function gotoPage(page, label) {
  const navLabel = page.locator("section[data-testid='stSidebar']").getByText(label, { exact: false });
  await navLabel.click();
  await page.waitForSelector(`text=${label}`, { timeout: 30000 });
  await page.waitForTimeout(1200);
}

async function clickButton(page, name) {
  const button = page.getByRole("button", { name });
  if ((await button.count()) > 0 && (await button.first().isVisible())) {
    await button.first().click();
    return true;
  }
  return false;
}

async function waitForText(page, labels, timeout = 90000) {
  await page.waitForFunction(
    (needles) => needles.some((needle) => document.body.innerText.includes(needle)),
    labels,
    { timeout },
  );
}

async function screenshot(page, filename) {
  await page.evaluate(() => window.scrollTo(0, 0));
  await page.waitForTimeout(1200);
  await page.screenshot({ path: path.join(outDir, filename), fullPage: false });
}

async function waitForChart(page, timeout = 90000) {
  await page.waitForSelector(".js-plotly-plot .main-svg", { timeout });
  await page.waitForTimeout(1500);
}

(async () => {
  const baseUrl = process.env.APP_URL || "http://127.0.0.1:8503";
  const browser = await chromium.launch({ headless: true });
  const page = await browser.newPage({ viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 });
  page.setDefaultTimeout(30000);

  await page.goto(baseUrl, { waitUntil: "domcontentloaded" });
  await waitForReady(page);

  await page.getByLabel("Asset / Ticker").fill("AAPL, MSFT, NVDA, SPY");
  await page.getByRole("button", { name: "Load Data" }).click();
  await waitForText(page, ["Price Chart", "Recent Performance", "Use Load Data to start."], 120000);
  await waitForChart(page);
  await screenshot(page, "overview.png");

  await gotoPage(page, "Data Explorer");
  await waitForText(page, ["Recent normalised OHLCV records", "Factor Snapshot"], 60000);
  await waitForChart(page);
  await screenshot(page, "data.png");

  await gotoPage(page, "Strategy Backtest");
  await waitForText(page, ["Strategy Backtest"], 60000);
  await clickButton(page, "Run backtest");
  await waitForText(page, ["Equity Curve", "Portfolio Drawdown", "Final value"], 90000);
  await waitForChart(page);
  await screenshot(page, "backtest.png");

  await gotoPage(page, "Risk Analysis");
  await waitForText(page, ["Rolling 63-Day Volatility", "Tail risk summary"], 60000);
  await waitForChart(page);
  await screenshot(page, "risk-analytics.png");

  await gotoPage(page, "Research Report");
  await waitForText(page, ["Research Report"], 60000);
  await clickButton(page, "Generate report");
  await waitForText(page, ["Executive Summary", "Price Performance Overview"], 90000);
  await screenshot(page, "ai-research-report.png");

  await browser.close();

  const captured = fs.readdirSync(outDir)
    .filter((file) => file.endsWith(".png"))
    .map((file) => ({ file, bytes: fs.statSync(path.join(outDir, file)).size }));
  console.log(JSON.stringify(captured, null, 2));
})();
