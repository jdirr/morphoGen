// The screenshots of the tour of the graphical interface (docs/tutorial_interface.md):
// a small study driven in the page of a running `morphogen gui`, a screenshot at each
// step. Run by docs/tutorials/make_screenshots.py, which starts the server:
//
//     node docs/tutorials/screenshots.js http://127.0.0.1:8765/ <folder>
//
// Needs Node.js and Playwright (with its Chromium).
const { chromium } = require('playwright');

(async () => {
  const [base, out] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 860 }, deviceScaleFactor: 1 });
  const errors = [];
  page.on('pageerror', (e) => errors.push(e.message));
  const shot = (name) => page.screenshot({ path: `${out}/interface_${name}.png` });
  const top = () => page.evaluate(() => window.scrollTo(0, 0));
  const set = (id, value) => page.evaluate(([i, v]) => {
    const el = document.getElementById(i);
    if (el.type === 'checkbox') el.checked = v; else el.value = v;
    el.dispatchEvent(new Event('input', { bubbles: true }));
    el.dispatchEvent(new Event('change', { bubbles: true }));
  }, [id, value]);
  const param = (name, value) => page.evaluate(([n, v]) => {
    const el = [...document.querySelectorAll('#gen-params [data-name]')].find((x) => x.dataset.name === n);
    el.value = v;
  }, [name, value]);

  // 1. generate a morphology: the menu of the models, the form, the sections and measures
  await page.goto(base + '#generate/boolean_spheres');
  await page.waitForSelector('#gen-list button.active');
  await param('radius', '1');
  await param('volume_fraction', '0.2');
  await set('gen-size', '10');
  await set('gen-seed', '1');
  await page.click('#gen-go');
  await page.waitForSelector('#gen-out h3', { timeout: 120000 });
  await page.waitForTimeout(2500); // the 3D view below the sections
  await top();
  await shot('generate');

  // 2. the search of the header: an option of a form
  await page.keyboard.press('/');
  await page.keyboard.type('ogden');
  await page.waitForSelector('#search-results .hit');
  await shot('search');
  await page.keyboard.press('Escape');

  // 3. characterize: the measures of the morphology against its model
  await page.click('#gen-out .sendto button[data-to="statistics"]');
  await page.waitForSelector('#tab-statistics.active');
  await page.click('#toolgo-statistics');
  await page.waitForSelector('#toolout-statistics table', { timeout: 300000 });
  await top();
  await shot('statistics');

  // 4. simulate: a loading path of the morphology, its matrix plastic
  await page.evaluate(() => { location.hash = '#mechanics/path'; });
  await page.waitForSelector('#tab-mechanics.active');
  await page.evaluate(() => {
    const source = document.getElementById('tool-path-source');
    source.value = [...source.options].find((o) => /boolean_spheres/.test(o.textContent)).value;
    source.dispatchEvent(new Event('change', { bubbles: true }));
  });
  if (await page.$('#tool-path-video')) await set('tool-path-video', false);
  await page.click('#toolgo-mechanics');
  await page.waitForSelector('#toolout-mechanics .views img', { timeout: 600000 });
  await page.setViewportSize({ width: 1280, height: 1240 }); // the values and the curves
  await page.waitForTimeout(500);
  await top();
  await shot('mechanics');
  await page.setViewportSize({ width: 1280, height: 860 });

  // 5. break it: the phase field on a planar section
  await page.evaluate(() => { location.hash = '#fracture/phase_field'; });
  await page.waitForSelector('#tab-fracture.active');
  await page.evaluate(() => {
    const source = document.getElementById('tool-phase_field-source');
    source.value = [...source.options].find((o) => /boolean_spheres/.test(o.textContent)).value;
    source.dispatchEvent(new Event('change', { bubbles: true }));
  });
  await set('tool-phase_field-voxels', '24');
  await page.click('#toolgo-fracture');
  await page.waitForSelector('#toolout-fracture .views img', { timeout: 900000 });
  await page.waitForTimeout(500);
  await top();
  await shot('fracture');

  // 6. export: the files of the morphology
  await page.evaluate(() => { location.hash = '#export'; });
  await page.waitForSelector('#tab-export.active');
  await page.evaluate(() => {
    const source = document.getElementById('exp-source');
    source.value = [...source.options].find((o) => /boolean_spheres/.test(o.textContent)).value;
    source.dispatchEvent(new Event('change', { bubbles: true }));
    for (const c of document.querySelectorAll('#exp-morphology [data-out]')) {
      c.checked = ['geometry', 'sections', 'level_set', 'surface'].includes(c.dataset.out);
    }
    document.getElementById('exp-shape').value = 64;
  });
  await page.click('#exp-go');
  await page.waitForSelector('#exp-out .files', { timeout: 300000 });
  await top();
  await shot('export');

  // 7. the map of the interface
  await page.click('#help-toggle');
  await page.waitForSelector('#help dl');
  await top();
  await shot('map');

  await browser.close();
  if (errors.length) {
    console.error('errors of the page:\n' + errors.join('\n'));
    process.exit(1);
  }
})();
