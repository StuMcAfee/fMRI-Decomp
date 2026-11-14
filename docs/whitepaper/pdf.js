const { chromium } = require('playwright');
const path = require('path');
(async () => {
  const b = await chromium.launch();
  const pg = await b.newPage();
  await pg.goto('file://' + path.join(__dirname, 'whitepaper.html'), { waitUntil: 'load', timeout: 60000 });
  await pg.waitForTimeout(1500);
  await pg.pdf({ path: path.join(__dirname, 'whitepaper.pdf'), format: 'Letter', printBackground: true,
    margin: { top: '16mm', bottom: '16mm', left: '14mm', right: '14mm' }, displayHeaderFooter: true,
    headerTemplate: '<span></span>',
    footerTemplate: '<div style="font:9px sans-serif;color:#8a8984;width:100%;text-align:center"><span class="pageNumber"></span> / <span class="totalPages"></span></div>' });
  await b.close();
})();
