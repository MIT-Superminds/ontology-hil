/**
 * json_to_docx.js
 * ---------------
 * Converts any JSON ontology hierarchy into a formatted Word document.
 * Works with two key formats:
 *
 *   FORMAT A — Process-ontology keys (with synsets and verb links):
 *     "label (synset.n.01, synset.n.02) - VerbLink (Verb.v.01)"
 *     "label (synset.n.01)"
 *
 *   FORMAT B — Plain hierarchy keys (no synsets):
 *     "Some Label"
 *     "[virtual] some_label (some_label.n.01)"
 *     "[How many?]"
 *     "[virtual] something"
 *
 *   The script auto-detects which format each key uses and renders accordingly.
 *
 * Usage:
 *   node json_to_docx.js [input.json] [output.docx] [title]
 *
 * Defaults:
 *   input  = process-ontology-mapped-v4.json
 *   output = (input filename with .docx extension)
 *   title  = (derived from input filename)
 *
 * Setup (run once in the SAME FOLDER as this script):
 *   npm install docx
 *
 * NOTE: Use local install (npm install docx), NOT global (npm install -g docx).
 * Global installs are not visible to scripts run with "node script.js".
 *
 * Heading levels assigned by depth in the JSON tree:
 *   depth 0  =>  Heading 1  (top-level keys of root object)
 *   depth 1  =>  Heading 2
 *   depth 2  =>  Heading 3
 *   depth 3+ =>  Bullet (indented proportionally, capped at 5 levels)
 *
 * Special key prefixes:
 *   [virtual]   =>  rendered in italics, slightly greyed
 *   [new]       =>  rendered with a blue badge
 *   [How...]    =>  rendered as a structural italic label, not a bullet
 *   [What...]   =>  same as above
 *
 * Trailing commas in JSON are fixed automatically before parsing.
 */

'use strict';

const fs   = require('fs');
const path = require('path');

// ─── Module loading with helpful error ────────────────────────────────────────
let docxModule;
try {
  docxModule = require('docx');
} catch (e) {
  console.error([
    '',
    'ERROR: Cannot find module "docx".',
    '',
    'Fix: run this command in the SAME FOLDER as json_to_docx.js:',
    '',
    '    npm install docx',
    '',
    'Do NOT use "npm install -g docx" -- global installs are not',
    'visible to scripts run with "node script.js".',
    '',
  ].join('\n'));
  process.exit(1);
}

const {
  Document, Packer, Paragraph, TextRun,
  HeadingLevel, AlignmentType, LevelFormat, BorderStyle,
} = docxModule;

// ─── CLI arguments ────────────────────────────────────────────────────────────
const inputFile  = process.argv[2] || 'actor-ontology-edited.json';
const outputFile = process.argv[3] || inputFile.replace(/\.json$/i, '.docx');
const titleArg   = process.argv[4] || null;

if (!fs.existsSync(inputFile)) {
  console.error('ERROR: Input file not found: ' + inputFile);
  process.exit(1);
}

// ─── Load and fix JSON ────────────────────────────────────────────────────────
let rawText = fs.readFileSync(inputFile, 'utf8');

// Remove BOM if present
rawText = rawText.replace(/^\uFEFF/, '');

// Fix trailing commas before } or ] (common mistake in hand-written JSON)
const fixedText = rawText.replace(/,(\s*[}\]])/g, '$1');

let data;
try {
  data = JSON.parse(fixedText);
} catch (e) {
  const lineMatch = e.message.match(/line (\d+)/);
  const lineInfo  = lineMatch ? ' (around line ' + lineMatch[1] + ')' : '';
  console.error('ERROR: Could not parse JSON' + lineInfo + ': ' + e.message);
  if (rawText !== fixedText) {
    console.error('(Trailing commas were automatically fixed; the remaining error is a different issue.)');
  }
  process.exit(1);
}

// ─── Derive document title ────────────────────────────────────────────────────
function titleFromFilename(f) {
  return path.basename(f, path.extname(f))
    .replace(/[-_]/g, ' ')
    .replace(/\b\w/g, function(c) { return c.toUpperCase(); });
}
const docTitle = titleArg || titleFromFilename(inputFile);

// ─── Constants ────────────────────────────────────────────────────────────────
const FONT          = 'Arial';
const COLOR_SYNSET  = '666666';  // grey for synset IDs
const COLOR_VERB    = '888888';  // lighter grey for verb link
const COLOR_VIRTUAL = '777777';  // grey for [virtual] nodes
const COLOR_NEW     = '2E75B6';  // blue for [new]
const COLOR_BRACKET = '555555';  // dark grey for [structural labels]

const HEADING_COLORS = { 0: '1F3864', 1: '2E4057', 2: '374151' };
const HEADING_SIZES  = { 0: 36,       1: 28,       2: 22       };
const HEADING_MAP    = {
  0: HeadingLevel.HEADING_1,
  1: HeadingLevel.HEADING_2,
  2: HeadingLevel.HEADING_3,
};

// ─── Key parser ───────────────────────────────────────────────────────────────
/**
 * Parse any node key into structured parts.
 * Returns { raw, isVirtual, isNew, isBracket, label, synsets, verbLink }
 *
 *   isVirtual  -- key starts with [virtual]
 *   isNew      -- key starts with [new]
 *   isBracket  -- whole label is a [structural bracket], no synsets
 *   label      -- display label (without [virtual]/[new] prefix)
 *   synsets    -- comma-separated synset string, or null
 *   verbLink   -- text after first " - ", or null
 */
function parseKey(raw) {
  var s = raw.trim();

  var isVirtual = s.indexOf('[virtual]') === 0;
  var isNew     = s.indexOf('[new]')     === 0;

  // Strip prefix for display
  var display = s;
  if (isVirtual) display = display.replace(/^\[virtual\]\s*/, '');
  if (isNew)     display = display.replace(/^\[new\]\s*/,     '');

  // Split off verb link at first " - "
  var dashIdx  = display.indexOf(' - ');
  var mainPart = dashIdx === -1 ? display                : display.slice(0, dashIdx);
  var verbLink = dashIdx === -1 ? null                   : display.slice(dashIdx + 3).trim();

  // Try to extract "label (synsets)" where synsets contain a .n. pattern
  var synsetsMatch = mainPart.match(/^(.*?)\s+\(([^)]*\.[nva]\.\d+[^)]*)\)\s*$/);
  var label, synsets;
  if (synsetsMatch) {
    label   = synsetsMatch[1].trim();
    synsets = synsetsMatch[2].trim();
  } else {
    label   = mainPart.trim();
    synsets = null;
  }

  // Detect purely structural bracket labels: e.g. "[How many?]", "[What kind?]"
  // The whole label (after prefix stripping) is wrapped in []
  var isBracket = !isVirtual && !isNew && !synsets && /^\[.*\]$/.test(label.trim());

  return { raw: raw, isVirtual: isVirtual, isNew: isNew,
           isBracket: isBracket, label: label, synsets: synsets, verbLink: verbLink };
}

// ─── Numbering config: 5 bullet levels ────────────────────────────────────────
var BULLET_CHARS = ['\u2022', '\u25E6', '\u2013', '\u00B7', '\u00B7'];
var INDENT_BASE  = 360;  // DXA per level

var NUMBERING_CONFIG = [{
  reference: 'bullets',
  levels: BULLET_CHARS.map(function(char, i) {
    return {
      level: i,
      format: LevelFormat.BULLET,
      text: char,
      alignment: AlignmentType.LEFT,
      style: {
        paragraph: {
          indent: { left: INDENT_BASE * (i + 1), hanging: INDENT_BASE },
          spacing: { after: 24, before: 0 },
        },
        run: { font: FONT, size: 20 },
      },
    };
  }),
}];

// ─── Paragraph builders ───────────────────────────────────────────────────────

function makeTitle(text) {
  return new Paragraph({
    spacing: { before: 0, after: 480 },
    children: [
      new TextRun({ text: text, font: FONT, size: 56, bold: true, color: '111827' }),
    ],
  });
}

function makeHeading(key, depth) {
  var parsed = parseKey(key);
  var color  = HEADING_COLORS[depth] || HEADING_COLORS[2];
  var size   = HEADING_SIZES[depth]  || 20;
  var level  = HEADING_MAP[depth]    || HeadingLevel.HEADING_3;
  var runs   = [];

  if (parsed.isNew) {
    runs.push(new TextRun({ text: '[new]  ', font: FONT, size: size,
                            bold: true, color: COLOR_NEW }));
  }
  if (parsed.isVirtual) {
    runs.push(new TextRun({ text: '[virtual]  ', font: FONT, size: size,
                            italics: true, color: COLOR_VIRTUAL }));
  }

  var labelText = parsed.label + (parsed.synsets ? '  (' + parsed.synsets + ')' : '');
  runs.push(new TextRun({
    text: labelText, font: FONT, size: size, bold: true,
    color: parsed.isNew ? COLOR_NEW : color,
  }));

  var borderColor = depth === 0 ? '1F3864' : (depth === 1 ? '4B5563' : 'D1D5DB');
  var borderSize  = depth === 0 ? 8        : (depth === 1 ? 4        : 2       );
  var spaceBefore = depth === 0 ? 520      : (depth === 1 ? 320      : 200     );

  return new Paragraph({
    heading: level,
    spacing: { before: spaceBefore, after: depth === 0 ? 120 : 60 },
    border: depth < 2 ? {
      bottom: { style: BorderStyle.SINGLE, size: borderSize,
                color: borderColor, space: 4 },
    } : undefined,
    children: runs,
  });
}

// Structural bracket label -- indented italic, not a bullet
function makeBracketLabel(text, bulletDepth) {
  var indent = INDENT_BASE * (bulletDepth + 1);
  return new Paragraph({
    spacing: { before: 120, after: 24 },
    indent: { left: indent },
    children: [
      new TextRun({ text: text, font: FONT, size: 18,
                    italics: true, color: COLOR_BRACKET }),
    ],
  });
}

// Bullet paragraph for a process/ontology node
function makeBullet(parsed, bulletDepth) {
  var level = Math.min(bulletDepth, 4);
  var runs  = [];

  if (parsed.isVirtual) {
    runs.push(new TextRun({ text: '[virtual]  ', font: FONT, size: 19,
                            italics: true, color: COLOR_VIRTUAL }));
  }
  if (parsed.isNew) {
    runs.push(new TextRun({ text: '[new]  ', font: FONT, size: 19,
                            bold: true, color: COLOR_NEW }));
  }

  // Main label
  runs.push(new TextRun({
    text: parsed.label,
    font: FONT, size: 20,
    bold:    !parsed.isVirtual,
    italics:  parsed.isVirtual,
    color: parsed.isVirtual ? COLOR_VIRTUAL : '111827',
  }));

  // Synsets in grey
  if (parsed.synsets) {
    runs.push(new TextRun({
      text: '  (' + parsed.synsets + ')',
      font: FONT, size: 18, color: COLOR_SYNSET,
    }));
  }

  // Verb link in grey italic with arrow
  if (parsed.verbLink) {
    runs.push(new TextRun({
      text: '    \u2192  ' + parsed.verbLink,
      font: FONT, size: 18, italics: true, color: COLOR_VERB,
    }));
  }

  return new Paragraph({
    numbering: { reference: 'bullets', level: level },
    spacing: { after: 28, before: 0 },
    children: runs,
  });
}

function spacer(before) {
  before = before || 100;
  return new Paragraph({
    spacing: { before: before, after: 0 },
    children: [new TextRun({ text: '', font: FONT })],
  });
}

// ─── Core recursive renderer ──────────────────────────────────────────────────
/**
 * Render a subtree recursively.
 *
 * headingDepth: 0/1/2 => render as H1/H2/H3; >= 3 => render as bullets
 * bulletDepth:  bullet indent level (only used when headingDepth >= 3)
 * out:          accumulator array of Paragraph objects
 */
function renderTree(obj, headingDepth, bulletDepth, out) {
  var entries = Object.entries(obj);
  for (var i = 0; i < entries.length; i++) {
    var key      = entries[i][0];
    var children = entries[i][1];
    var parsed   = parseKey(key);
    var hasKids  = children && typeof children === 'object'
                   && Object.keys(children).length > 0;

    if (headingDepth <= 2) {
      // ── Heading mode ──────────────────────────────────────────────────────
      if (parsed.isBracket) {
        // Structural bracket at heading level: render as indented label, not heading
        out.push(makeBracketLabel(parsed.label, 0));
      } else {
        out.push(makeHeading(key, headingDepth));
      }
      if (hasKids) {
        renderTree(children, headingDepth + 1, 0, out);
      }
    } else {
      // ── Bullet mode ───────────────────────────────────────────────────────
      if (parsed.isBracket) {
        out.push(makeBracketLabel(parsed.label, bulletDepth));
      } else {
        out.push(makeBullet(parsed, bulletDepth));
      }
      if (hasKids) {
        renderTree(children, headingDepth, bulletDepth + 1, out);
      }
    }
  }
}

// ─── Determine render root ────────────────────────────────────────────────────
// If the root has exactly one key that is just a wrapper (e.g. {"process":{...}}),
// unwrap it so we don't waste H1 on a single wrapper node.
function getRenderRoot(data) {
  var keys = Object.keys(data);
  if (keys.length === 1 && data[keys[0]] && typeof data[keys[0]] === 'object') {
    var inner = data[keys[0]];
    if (Object.keys(inner).length > 0) {
      return inner;
    }
  }
  return data;
}

// ─── Build document ───────────────────────────────────────────────────────────
var paragraphs = [];

paragraphs.push(makeTitle(docTitle));

// Show legend only when verb links are present
if (fixedText.indexOf(' - ') !== -1) {
  paragraphs.push(new Paragraph({
    spacing: { after: 240 },
    children: [
      new TextRun({ text: 'Each entry:  ', font: FONT, size: 19, color: '444444' }),
      new TextRun({ text: 'label  ',       font: FONT, size: 19, bold: true }),
      new TextRun({ text: '(synset.n.xx)  ', font: FONT, size: 19, color: COLOR_SYNSET }),
      new TextRun({ text: '\u2192  Verb Ontology Link (Verb.v.xx)',
                    font: FONT, size: 19, italics: true, color: COLOR_VERB }),
    ],
  }));
}

var renderRoot = getRenderRoot(data);
renderTree(renderRoot, 0, 0, paragraphs);
paragraphs.push(spacer(200));

// ─── Styles ───────────────────────────────────────────────────────────────────
var styles = {
  default: { document: { run: { font: FONT, size: 20 } } },
  paragraphStyles: [
    { id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal',
      quickFormat: true,
      run:       { size: 36, bold: true, font: FONT },
      paragraph: { spacing: { before: 520, after: 120 }, outlineLevel: 0 } },
    { id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal',
      quickFormat: true,
      run:       { size: 28, bold: true, font: FONT },
      paragraph: { spacing: { before: 320, after: 60  }, outlineLevel: 1 } },
    { id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal',
      quickFormat: true,
      run:       { size: 22, bold: true, font: FONT },
      paragraph: { spacing: { before: 200, after: 40  }, outlineLevel: 2 } },
  ],
};

// ─── Write output ─────────────────────────────────────────────────────────────
var doc = new Document({
  numbering: { config: NUMBERING_CONFIG },
  styles: styles,
  sections: [{
    properties: {
      page: {
        size:   { width: 12240, height: 15840 },
        margin: { top: 1080, right: 1080, bottom: 1080, left: 1260 },
      },
    },
    children: paragraphs,
  }],
});

Packer.toBuffer(doc)
  .then(function(buf) {
    fs.writeFileSync(outputFile, buf);
    console.log('Written: ' + outputFile + '  (' + Math.round(buf.length / 1024) + ' KB)');
  })
  .catch(function(err) {
    console.error('Error building document: ' + (err.message || err));
    process.exit(1);
  });
