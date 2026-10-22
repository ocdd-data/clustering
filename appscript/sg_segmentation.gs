/**
 * SG user base segmentation updater (Apps Script, bound to "SG User Base Segmentation").
 *
 * On the 1st of each month monthly/rider_clustering_m.py posts, in the Slack thread of its report,
 * segmentation_sheet_SG_<Mon YYYY>.csv (utils/sheet_export.py): columns A:D are section / from / to / value
 * for the month's riders per cluster (and total), the transition count matrix and the transition
 * percentage matrix; F1:G2 tag it: report = SG Segmentation, month_start = the month's first day.
 *
 * To fill a month (e.g. Sep 2026): File > Import > Upload > "Insert new sheet(s)". The script reads the
 * tags first and only goes ahead for a SG Segmentation file; anything else is refused and nothing changes.
 * Then:
 *   Matrices (Absolute) / (Rate)  the month's group ("Aug-Sep 2026": last month's clusters 0-4 down,
 *                                 this month's clusters 0-4 and Churned across) gets the count / percentage
 *                                 matrix; the X (didn't ride last month) row of the files is not used
 *   Rider(New)                    the month's column: riders per cluster and Total, churned riders per
 *                                 cluster (the counts' X column), and churn rate as formulas on Matrices
 *                                 (Rate)'s Churned column / 100
 * Each year's matrices sit in their own table, newest on top. When the Dec-Jan group is filled (the January
 * file), next year's tables are added above it, labelled and empty; Rider(New) gets next year's 12 columns
 * when a January file arrives. Labels are checked before anything is written. Re-importing a month
 * overwrites it. Rider(New)'s colour rules are widened to each month filled. On success the imported tab is deleted; on an error it stays, with the reason in red at F4.
 *
 * Setup: Extensions > Apps Script, paste this file, save, reload the sheet, Segmentation > Enable auto-update.
 */

const REPORT = 'SG Segmentation';
const RIDERS_TAB = 'Rider(New)';
const MATRIX_TABS = { counts: 'Matrices (Absolute)', percents: 'Matrices (Rate)' };
const NAMES = ['New or Inactive Riders', 'Occasional Riders', 'Moderately Active Riders', 'Regular Riders', 'Highly Active Riders'];
const TO = ['0', '1', '2', '3', '4', 'X'];        // a group's 6 columns: clusters 0-4, then Churned
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

// Rider(New): year labels in row 1 (merged across their months), month names in row 2, labels in column B.
const R = { YEAR_ROW: 1, MONTH_ROW: 2, LABEL_COL: 2, TOTALS: 3, TOTAL_ROW: 8, CHURN: 10, RATE: 16 };
// Matrices: per year a table of title row (year), group labels ("Jan-Feb 2026"), headers, 5 data rows; tables
// 9 rows apart, newest on top. Groups are 6 columns wide from column C, Jan-Feb first, Dec-Jan last.
const M = { FIRST_COL: 3, GROUP: 6, TABLE_ROWS: 8, GAP: 1, LABEL_COL: 2 };

const REPORT_KEY = 'report';
const MONTH_KEY = 'month_start';
const TAG_SCAN_COLS = 30;

// ---------- menu and triggers ----------

function onOpen() {
  SpreadsheetApp.getUi().createMenu('Segmentation')
    .addItem('Update now', 'updateNow')
    .addItem('Fill a specific month…', 'fillSpecificMonth')
    .addItem('Remove a month…', 'removeMonth')
    .addItem('Enable auto-update', 'enableAutoUpdate')
    .addToUi();
}

function updateNow() { run_(true, null); }

function fillSpecificMonth() {
  const month = askMonth_('Fill a specific month', "Month, e.g. 2026-09. That month's cells are overwritten.");
  if (month) run_(true, month);
}

// Installable onChange: runs on File > Import and on edits, and returns at once unless a tagged file has
// been imported.
function onChangeHandler(e) { run_(false, null); }

function enableAutoUpdate() {
  const ss = SpreadsheetApp.getActive();
  ScriptApp.getProjectTriggers()
    .filter(t => t.getHandlerFunction() === 'onChangeHandler')
    .forEach(t => ScriptApp.deleteTrigger(t));
  ScriptApp.newTrigger('onChangeHandler').forSpreadsheet(ss).onChange().create();
  ss.toast(`Auto-update is on: import a ${REPORT} file as new sheets and the month fills itself.`, REPORT, 8);
}

function askMonth_(title, text) {
  const ui = SpreadsheetApp.getUi();
  const r = ui.prompt(title, text, ui.ButtonSet.OK_CANCEL);
  if (r.getSelectedButton() !== ui.Button.OK) return null;
  const month = r.getResponseText().trim();
  if (!/^\d{4}-\d{2}$/.test(month)) { ui.alert(`"${month}" is not a month like 2026-09.`); return null; }
  return month;
}

// ---------- run ----------

function run_(manual, monthOverride) {
  const ss = SpreadsheetApp.getActive();
  const lock = LockService.getDocumentLock();
  if (!lock.tryLock(manual ? 30000 : 1000)) return;
  let imp = null;
  try {
    const found = findImports_(ss);
    if (!found.length) {
      if (manual) throw new Error(`No tagged file found. Import segmentation_sheet_SG_<month>.csv with File > Import > Upload > "Insert new sheet(s)".`);
      return;
    }
    imp = found[0];
    // A failed import stays with its error noted; rerun it from the menu, not on every later edit.
    if (!manual && String(errorCell_(imp).getValue()).indexOf('Not filled') === 0) return;
    if (found.length > 1)
      throw new Error(`${found.length} imported files are in the sheet (${found.map(f => f.sheet.getName()).join(', ')}). Delete the ones you don't need, then use Segmentation > Update now.`);
    if (imp.report !== REPORT)
      throw new Error(`This is a ${imp.report} file, but this spreadsheet takes ${REPORT}. Nothing was changed; delete the imported tab.`);

    const tz = ss.getSpreadsheetTimeZone() || Session.getScriptTimeZone() || 'Asia/Singapore';
    const month = monthOverride || monthKey_(imp.monthValue, tz);
    const src = readSource_(imp.sheet);
    const period = prevMonth_(month);                       // the matrices group is last month -> this month
    const pYear = Number(period.slice(0, 4)), pIdx = Number(period.slice(5)) - 1;

    // Everything is checked before anything is written.
    const riders = ss.getSheetByName(RIDERS_TAB);
    if (!riders) throw new Error(`Tab "${RIDERS_TAB}" not found.`);
    checkRiderLabels_(riders);
    const mc = monthCols_(riders);
    const riderCol = mc.m[month] || null;
    if (!riderCol && !(month === nextMonth_(mc.lastKey) && month.slice(5) === '01'))
      throw new Error(`${RIDERS_TAB} has no ${month} column, and it isn't the January after the last year (${mc.lastKey}).`);
    const tabs = {};
    Object.keys(MATRIX_TABS).forEach(key => {
      const sh = ss.getSheetByName(MATRIX_TABS[key]);
      if (!sh) throw new Error(`Tab "${MATRIX_TABS[key]}" not found.`);
      const tables = findTables_(sh);
      if (!tables[pYear] && pYear !== newestYear_(tables) + 1)
        throw new Error(`${MATRIX_TABS[key]} has no ${pYear} table, and ${pYear} isn't the year after the newest (${newestYear_(tables)}).`);
      if (tables[pYear]) checkTable_(sh, tables[pYear], pYear);
      tabs[key] = sh;
    });

    // Matrices: the month's group in both tabs (adding the year's tables first if needed).
    let churnCell = null;
    Object.keys(MATRIX_TABS).forEach(key => {
      const sh = tabs[key];
      let tables = findTables_(sh);
      if (!tables[pYear]) { addYearTable_(sh, pYear); tables = findTables_(sh); }
      const top = tables[pYear];
      const col = M.FIRST_COL + M.GROUP * pIdx;
      sh.getRange(top + 3, col, 5, 6).setValues(TO.slice(0, 5).map(a => TO.map(b => src[key][a][b])));
      if (key === 'percents') churnCell = { col: col + 5, row: top + 3 };
    });

    // Rider(New): the month's column (adding next year's columns in January).
    const col = riderCol || addYearColumns_(riders, mc, Number(month.slice(0, 4)));
    const rateCol = colLetter_(churnCell.col);
    riders.getRange(R.TOTALS, col, 5, 1).setValues(TO.slice(0, 5).map(k => [src.riders[k]]));
    riders.getRange(R.TOTAL_ROW, col).setValue(src.riders.total);
    riders.getRange(R.CHURN, col, 5, 1).setValues(TO.slice(0, 5).map(k => [src.counts[k].X]));
    riders.getRange(R.RATE, col, 5, 1).setFormulas(TO.slice(0, 5).map((_, i) =>
      [`='${MATRIX_TABS.percents}'!${rateCol}${churnCell.row + i}/100`]));
    extendFormatting_(riders, col);

    // December filled (the Dec-Jan group): next year's tables, ready for February.
    if (pIdx === 11) Object.values(tabs).forEach(sh => { if (!findTables_(sh)[pYear + 1]) addYearTable_(sh, pYear + 1); });
    SpreadsheetApp.flush();

    ss.deleteSheet(imp.sheet);
    const msg = `Filled ${month} (matrices group ${MONTHS[pIdx]}-${MONTHS[(pIdx + 1) % 12]} ${pYear}).` + (pIdx === 11 ? ` Added the ${pYear + 1} matrices tables.` : '');
    if (manual) SpreadsheetApp.getUi().alert(msg);
    else ss.toast(msg, REPORT, 15);
  } catch (err) {
    const msg = 'Not filled: ' + err.message;
    if (imp) errorCell_(imp).setValue(msg).setFontColor('#c5221f').setFontWeight('bold');
    if (manual) SpreadsheetApp.getUi().alert(msg);
    else ss.toast(msg, REPORT, 30);
  } finally {
    lock.releaseLock();
  }
}

// ---------- imported file ----------

// Tabs whose row 1 holds a "report" tag: each is one imported file.
function findImports_(ss) {
  const found = [];
  ss.getSheets().forEach(sh => {
    const width = Math.min(sh.getLastColumn(), TAG_SCAN_COLS);
    if (width < 2) return;
    const rows = sh.getRange(1, 1, 2, width).getValues();
    const col = rows[0].findIndex(v => String(v).trim() === REPORT_KEY);
    if (col < 0 || col + 1 >= width) return;
    found.push({ sheet: sh, tagCol: col + 1, report: String(rows[0][col + 1]).trim(), monthValue: rows[1][col + 1] });
  });
  return found;
}

function errorCell_(imp) { return imp.sheet.getRange(4, imp.tagCol); }

function monthKey_(v, tz) {
  const s = v instanceof Date ? Utilities.formatDate(new Date(v.getTime() + 12 * 3600 * 1000), tz, 'yyyy-MM-dd') : String(v).trim();
  if (!/^\d{4}-\d{2}(-\d{2})?$/.test(s))
    throw new Error(`The file's ${MONTH_KEY} tag should be a date like 2026-09-01, got "${v}".`);
  return s.slice(0, 7);
}

// A CSV import keeps numbers longer than 15 digits as text; turn anything that reads as a number into one.
function toNumber_(v) {
  return typeof v === 'string' && /^\s*-?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$/.test(v) ? Number(v) : v;
}

// { riders: {0..4, total}, counts: {from: {to: n}}, percents: {from: {to: %}} } from columns A:D.
function readSource_(sh) {
  const out = { riders: {}, counts: {}, percents: {} };
  sh.getRange(2, 1, Math.max(sh.getLastRow() - 1, 1), 4).getValues().forEach(([section, from, to, value]) => {
    section = String(section).trim(); from = String(from).trim(); to = String(to).trim();
    const v = toNumber_(value);
    if (section === 'riders') out.riders[from] = v;
    else if (section in out) (out[section][from] = out[section][from] || {})[to] = v;
  });
  const missing = [];
  ['0', '1', '2', '3', '4', 'total'].forEach(k => { if (typeof out.riders[k] !== 'number') missing.push(`riders ${k}`); });
  ['counts', 'percents'].forEach(s => TO.slice(0, 5).forEach(a => TO.forEach(b => {
    if (typeof (out[s][a] || {})[b] !== 'number') missing.push(`${s} ${a}->${b}`);
  })));
  if (missing.length) throw new Error(`The imported file is missing or has non-numbers for: ${missing.slice(0, 6).join(', ')}${missing.length > 6 ? ', …' : ''}.`);
  return out;
}

// ---------- helpers ----------

function norm_(s) { return String(s).replace(/\s+/g, ' ').trim().toLowerCase(); }
function colLetter_(c) { let s = ''; for (; c > 0; c = Math.floor((c - 1) / 26)) s = String.fromCharCode(65 + (c - 1) % 26) + s; return s; }
function prevMonth_(key) { let [y, m] = key.split('-').map(Number); if (--m < 1) { m = 12; y--; } return `${y}-${String(m).padStart(2, '0')}`; }
function nextMonth_(key) { let [y, m] = key.split('-').map(Number); if (++m > 12) { m = 1; y++; } return `${y}-${String(m).padStart(2, '0')}`; }

// ---------- Rider(New) ----------

function checkRiderLabels_(sh) {
  const labels = sh.getRange(1, R.LABEL_COL, R.RATE + 4, 1).getValues().map(r => norm_(r[0]));
  const want = [[R.TOTALS, NAMES], [R.TOTAL_ROW, ['Total']], [R.CHURN, NAMES], [R.RATE, NAMES]];
  want.forEach(([first, names]) => names.forEach((n, i) => {
    if (labels[first + i - 1] !== norm_(n))
      throw new Error(`${RIDERS_TAB} B${first + i} should be "${n}" but is "${labels[first + i - 1]}". Has a row been added or removed?`);
  }));
}

// 'YYYY-MM' -> column from the month names (row 2); the year starts at the first year label in row 1 and
// steps up each time the months wrap round (Dec -> Jan).
function monthCols_(sh) {
  const width = sh.getLastColumn();
  const names = sh.getRange(R.MONTH_ROW, 1, 1, width).getValues()[0];
  const labels = sh.getRange(R.YEAR_ROW, 1, 1, width).getValues()[0];
  const m = {};
  let year = null, prev = 99, lastKey = null, lastCol = null;
  names.forEach((v, i) => {
    const mi = MONTHS.indexOf(String(v).trim().slice(0, 3));
    if (mi < 0 || typeof v !== 'string') return;
    if (year === null) {
      const j = labels.findIndex((x, k) => k >= i && typeof x === 'number' && x > 2000);
      if (j < 0) return;
      year = labels[j];
    } else if (mi <= prev) year++;
    prev = mi;
    const k = `${year}-${String(mi + 1).padStart(2, '0')}`;
    m[k] = i + 1; lastKey = k; lastCol = i + 1;
  });
  return { m, lastKey, lastCol };
}

// Next year's 12 columns after the last December: formats from the year before, year label, month names.
function addYearColumns_(sh, mc, year) {
  const last = mc.lastCol, first = last + 1;
  sh.insertColumnsAfter(last, 12);
  const height = R.RATE + 4;
  sh.getRange(1, last - 11, height, 12).copyTo(sh.getRange(1, first, height, 12), SpreadsheetApp.CopyPasteType.PASTE_FORMAT, false);
  for (let i = 0; i < 12; i++) sh.setColumnWidth(first + i, sh.getColumnWidth(last - 11 + i));
  const label = sh.getRange(R.YEAR_ROW, first, 1, 12);
  if (!sh.getRange(R.YEAR_ROW, first).isPartOfMerge()) label.merge();
  sh.getRange(R.YEAR_ROW, first).setValue(year);
  sh.getRange(R.MONTH_ROW, first, 1, 12).setValues([MONTHS]);
  return first;
}

// Colour rules on Rider(New) that end at the column before this month are widened to cover it, so the month
// is coloured like the months before (the rules themselves are unchanged).
function extendFormatting_(sh, col) {
  const rules = sh.getConditionalFormatRules();
  let changed = false;
  const out = rules.map(rule => {
    const ranges = rule.getRanges();
    const widened = ranges.map(r => (r.getLastColumn() === col - 1 && r.getColumn() < col
      ? sh.getRange(r.getRow(), r.getColumn(), r.getNumRows(), col - r.getColumn() + 1) : r));
    if (widened.every((r, i) => r === ranges[i])) return rule;
    changed = true;
    return rule.copy().setRanges(widened).build();
  });
  if (changed) sh.setConditionalFormatRules(out);
}

// ---------- Matrices ----------

// year -> the table's first (title) row, found from its first group label ("Jan-Feb 2026") in column C.
function findTables_(sh) {
  const col = sh.getRange(1, M.FIRST_COL, sh.getLastRow(), 1).getValues();
  const out = {};
  col.forEach((r, i) => { const m = /^Jan-Feb (\d{4})$/.exec(String(r[0]).trim()); if (m) out[Number(m[1])] = i; });
  return out;  // label row is i + 1, so the title row is i (1-based)
}

function newestYear_(tables) { return Math.max(...Object.keys(tables).map(Number)); }

function checkTable_(sh, top, year) {
  const labels = sh.getRange(top + 3, M.LABEL_COL, 5, 1).getValues().map(r => norm_(r[0]));
  NAMES.forEach((n, i) => {
    if (labels[i] !== norm_(n)) throw new Error(`${sh.getName()} ${year} table row ${top + 3 + i} should be "${n}" but is "${labels[i]}".`);
  });
  const head = sh.getRange(top + 2, M.FIRST_COL, 1, 6).getValues()[0].map(norm_);
  NAMES.concat(['Churned']).forEach((n, i) => {
    if (head[i] !== norm_(n)) throw new Error(`${sh.getName()} ${year} table header ${colLetter_(M.FIRST_COL + i)}${top + 2} should be "${n}" but is "${head[i]}".`);
  });
}

// A new year's table on top: rows inserted above the newest table, which is copied with the empty row
// below it (labels, formats, merges; the copied empty row keeps the gap plain rather than taking the title
// band's colour), relabelled for the year and emptied.
function addYearTable_(sh, year) {
  const tables = findTables_(sh);
  const newest = tables[newestYear_(tables)];
  const width = sh.getLastColumn();
  const add = M.TABLE_ROWS + M.GAP;
  sh.insertRowsBefore(newest, add);
  const top = newest, tpl = newest + add;
  sh.getRange(tpl, 1, add, width).copyTo(sh.getRange(top, 1, add, width), SpreadsheetApp.CopyPasteType.PASTE_NORMAL, false);
  sh.getRange(top, M.LABEL_COL, 1, 2).setValues([[year, year]]);
  const labels = MONTHS.map((m, k) => (k < 11 ? `${m}-${MONTHS[k + 1]} ${year}` : `Dec ${year}-Jan ${year + 1}`));
  labels.forEach((l, k) => sh.getRange(top + 1, M.FIRST_COL + M.GROUP * k).setValue(l));
  sh.getRange(top + 3, M.FIRST_COL, 5, M.GROUP * 12).clearContent();
}

// ---------- undo ----------

function removeMonth() {
  const ui = SpreadsheetApp.getUi();
  const month = askMonth_('Remove a month', 'Month, e.g. 2026-09. Clears its Rider(New) column and its matrices group.');
  if (!month) return;
  const ss = SpreadsheetApp.getActive();
  const period = prevMonth_(month), pYear = Number(period.slice(0, 4)), pIdx = Number(period.slice(5)) - 1;
  const riders = ss.getSheetByName(RIDERS_TAB);
  const col = monthCols_(riders).m[month];
  const ok = ui.alert(`Remove ${month}?`, `Clears ${RIDERS_TAB}'s ${month} column and the ${MONTHS[pIdx]}-${MONTHS[(pIdx + 1) % 12]} ${pYear} group in both matrices tabs.`, ui.ButtonSet.YES_NO);
  if (ok !== ui.Button.YES) return;
  if (col) [[R.TOTALS, 6], [R.CHURN, 5], [R.RATE, 5]].forEach(([r, n]) => riders.getRange(r, col, n, 1).clearContent());
  Object.values(MATRIX_TABS).forEach(name => {
    const sh = ss.getSheetByName(name), top = findTables_(sh)[pYear];
    if (top) sh.getRange(top + 3, M.FIRST_COL + M.GROUP * pIdx, 5, M.GROUP).clearContent();
  });
  ui.alert(`Removed ${month}.`);
}
