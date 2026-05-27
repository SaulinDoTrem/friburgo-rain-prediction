const fs = require("fs");
const path = require("path");
const { chromium } = require("playwright");

const OUTPUT_DIR = __dirname;
const BASE_URL = "https://tempo.inmet.gov.br/TabelaEstacoes/A001";
const HEADLESS = true;
const PAGE_START_DELAY_MS = Number(4000);
const TYPE_DELAY_MS = Number(50);

const MIN_DATE = new Date("2010-09-20T00:00:00");
const MAX_DATE = new Date("2026-05-26T00:00:00");

function pad(n) {
    return String(n).padStart(2, "0");
}

function formatUiDate(date) {
    return `${pad(date.getDate())}/${pad(date.getMonth() + 1)}/${date.getFullYear()}`;
}

function formatDigitsDate(date) {
    return `${pad(date.getDate())}${pad(date.getMonth() + 1)}${date.getFullYear()}`;
}

function formatIsoDate(date) {
    return `${date.getFullYear()}-${pad(date.getMonth() + 1)}-${pad(date.getDate())}`;
}

function endOfMonth(year, monthIdx) {
    return new Date(year, monthIdx + 1, 0);
}

function minDate(a, b) {
    return a <= b ? a : b;
}

function sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
}

function buildPeriods(minDateLimit, maxDateLimit) {
    const periods = [];

    for (
        let year = minDateLimit.getFullYear();
        year <= maxDateLimit.getFullYear();
        year += 1
    ) {
        const candidates = [
            {
                start: new Date(year, 0, 1),
                end: endOfMonth(year, 4), // 31/05
            },
            {
                start: new Date(year, 5, 1),
                end: endOfMonth(year, 11), // 31/12
            },
        ];

        for (const candidate of candidates) {
            const start =
                candidate.start < minDateLimit ? minDateLimit : candidate.start;
            const end = minDate(candidate.end, maxDateLimit);

            if (
                start <= end &&
                !(candidate.end < minDateLimit) &&
                !(candidate.start > maxDateLimit)
            ) {
                periods.push({ start: new Date(start), end: new Date(end) });
            }
        }
    }

    return periods;
}

async function typeDateDigitsByXPath(page, xpath, dateValue) {
    const locator = page.locator(`xpath=${xpath}`);
    await locator.waitFor({ timeout: 30000 });
    await locator.click({ timeout: 30000 });
    await locator.press("ArrowLeft");
    await locator.press("ArrowLeft");
    await locator.type(dateValue, { delay: TYPE_DELAY_MS });
    await locator.blur();
}

async function configureStation(page) {
    await page.waitForLoadState("domcontentloaded");

    // 1 - clica em .left.menu > .bars.icon.header-icon
    await page
        .locator(".left.menu > .bars.icon.header-icon")
        .click({ timeout: 30000 });

    // 2 - clica no elemento indicado
    await page
        .locator("xpath=/html/body/div[1]/div[2]/div[1]/div[2]/div[3]")
        .click({ timeout: 30000 });

    // 3 - clica no item da estacao
    await page
        .locator(
            "xpath=/html/body/div[1]/div[2]/div[1]/div[2]/div[3]/div[2]/div[417]/span",
        )
        .click({ timeout: 30000 });
}

async function downloadForPeriod(page, period) {
    const startUi = formatUiDate(period.start);
    const endUi = formatUiDate(period.end);
    const startDigits = formatDigitsDate(period.start);
    const endDigits = formatDigitsDate(period.end);
    const startIso = formatIsoDate(period.start);
    const endIso = formatIsoDate(period.end);
    const outName = `estacao-salinas-${startIso}-${endIso}.csv`;
    const outPath = path.join(OUTPUT_DIR, 'data', outName);

    if (fs.existsSync(outPath)) {
        console.log(`Ja existe, pulando: ${outName}`);
        return;
    }

    await configureStation(page);

    console.log(`Baixando periodo ${startUi} -> ${endUi}`);

    // 4 - muda a data inicial
    await typeDateDigitsByXPath(
        page,
        "/html/body/div[1]/div[2]/div[1]/div[2]/div[4]/input",
        startDigits,
    );

    // 5 - muda a data final
    await typeDateDigitsByXPath(
        page,
        "/html/body/div[1]/div[2]/div[1]/div[2]/div[5]/input",
        endDigits,
    );

    // 6 - clica no botao de consulta
    await page
        .locator("xpath=/html/body/div[1]/div[2]/div[1]/div[2]/button")
        .click({ timeout: 30000 });

    try {
        // 7 - espera o link de download e clica
        const downloadLink = page.locator(
            "xpath=/html/body/div[1]/div[2]/div[2]/div/div/div/span/a",
        );
        await downloadLink.waitFor({ state: "visible", timeout: 60000 });

        try {
            const [download] = await Promise.all([
                page.waitForEvent("download", { timeout: 30000 }),
                downloadLink.click({ timeout: 30000 }),
            ]);

            await download.saveAs(outPath);
            console.log(`OK: ${outName}`);
            return;
        } catch (_) {
            // fallback: alguns navegadores/sessões expõem apenas href sem evento de download
        }

        const href = await downloadLink.getAttribute("href");
        if (!href) {
            console.log(
                `Sem download para ${startUi} -> ${endUi}: link sem href`,
            );
            return;
        }

        const response = await page.request.get(href, { timeout: 60000 });
        if (!response.ok()) {
            console.log(
                `Sem download para ${startUi} -> ${endUi}: HTTP ${response.status()}`,
            );
            return;
        }

        const data = await response.body();
        fs.writeFileSync(outPath, data);
        console.log(`OK (fallback): ${outName}`);
    } catch (err) {
        console.log(`Sem download para ${startUi} -> ${endUi}: ${err.message}`);
    }
}

(async () => {
    const periods = buildPeriods(MIN_DATE, MAX_DATE);
    console.log(`Total de periodos: ${periods.length}`);
    console.log(`Modo headless: ${HEADLESS}`);
    console.log(`Espera inicial da pagina (ms): ${PAGE_START_DELAY_MS}`);

    const browser = await chromium.launch({
        headless: HEADLESS,
    });
    const context = await browser.newContext({ acceptDownloads: true });
    const page = await context.newPage();

    try {
        await page.goto(BASE_URL, {
            waitUntil: "domcontentloaded",
            timeout: 120000,
        });
        await sleep(PAGE_START_DELAY_MS);
        
        for (const period of periods) {
            await downloadForPeriod(page, period);
        }
    } finally {
        await context.close();
        await browser.close();
    }
})();
