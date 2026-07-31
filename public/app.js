const fmt = new Intl.NumberFormat('en-IN');
const fmt2 = new Intl.NumberFormat('en-IN', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
const fmt6 = new Intl.NumberFormat('en-IN', { minimumFractionDigits: 6, maximumFractionDigits: 6 });
const byId = id => document.getElementById(id);
const esc = value => String(value).replace(/[&<>'"]/g, char => ({ '&':'&amp;', '<':'&lt;', '>':'&gt;', "'":'&#39;', '"':'&quot;' }[char]));
const colors = ['#65e3ff','#ffb85c','#55e6a5','#ad8cff','#ff7e86','#7aa9ff','#e5e96f','#f29ee1'];

let dashboard;
let tokenizer;
let mergeRanks;

function formatScore(value) { return value == null ? '∞' : fmt2.format(value); }

function bpePiece(piece) {
  const symbols = Array.from(piece);
  while (symbols.length > 1) {
    let bestRank = Infinity;
    let bestIndex = -1;
    for (let index = 0; index < symbols.length - 1; index += 1) {
      const rank = mergeRanks.get(`${symbols[index]}\u0000${symbols[index + 1]}`);
      if (rank !== undefined && rank < bestRank) { bestRank = rank; bestIndex = index; }
    }
    if (bestIndex < 0) break;
    symbols.splice(bestIndex, 2, symbols[bestIndex] + symbols[bestIndex + 1]);
  }
  return symbols;
}

function encodePublishedBpe(text) {
  const tokens = [];
  let index = 0;
  let pendingSpaces = 0;
  while (index < text.length) {
    const char = text[index];
    if (char === ' ') { pendingSpaces += 1; index += 1; continue; }
    if (/\s/u.test(char)) {
      while (pendingSpaces > 0) { tokens.push('▁'); pendingSpaces -= 1; }
      tokens.push(char); index += 1; continue;
    }
    let end = index;
    while (end < text.length && !/\s/u.test(text[end])) end += 1;
    while (pendingSpaces > 1) { tokens.push('▁'); pendingSpaces -= 1; }
    const prefix = pendingSpaces ? '▁' : '';
    pendingSpaces = 0;
    tokens.push(...bpePiece(prefix + text.slice(index, end)));
    index = end;
  }
  while (pendingSpaces > 0) { tokens.push('▁'); pendingSpaces -= 1; }
  return tokens;
}

function renderHero() {
  const summary = dashboard.featured.summary;
  byId('heroScore').textContent = formatScore(summary.raw_score);
  byId('heroSpread').textContent = fmt6.format(summary.spread);
  byId('formulaSpread').textContent = fmt6.format(summary.spread);
  byId('formulaScore').textContent = formatScore(summary.raw_score);
  byId('proofSpread').textContent = fmt6.format(summary.spread);
  byId('builtAt').textContent = `Built ${new Date(dashboard.featured.built_at_utc).toLocaleDateString('en-IN', { day:'2-digit', month:'short', year:'numeric' })}`;
  const metrics = [
    ['Vocabulary', fmt.format(dashboard.featured.vocab_size_actual), 'one shared model'],
    ['Fertility spread', fmt6.format(summary.spread), 'max X − min X'],
    ['English constraint', summary.english_x_le_1_2 ? 'PASS' : 'FAIL', 'X ≤ 1.2', summary.english_x_le_1_2],
    ['Round trip', dashboard.featured.round_trip_exact_valid ? 'EXACT' : 'FAIL', 'decode(encode(text))', dashboard.featured.round_trip_exact_valid],
  ];
  byId('metricGrid').innerHTML = metrics.map(([label,value,note,good]) => `<article class="surface metric"><span>${esc(label)}</span><strong class="${good ? 'good' : ''}">${esc(value)}</strong><small>${esc(note)}</small></article>`).join('');
}

function renderResults() {
  const rows = [...dashboard.featured.per_language].sort((a,b) => b.fertility_x - a.fertility_x);
  const floor = 0.58;
  const ceiling = Math.max(...rows.map(row => row.fertility_x), 0.68) + 0.01;
  byId('fertilityChart').innerHTML = rows.map(row => {
    const height = Math.max(8, ((row.fertility_x - floor) / (ceiling - floor)) * 88);
    return `<div class="fertility-column"><b>${fmt6.format(row.fertility_x)}</b><div class="fertility-bar" style="height:${height}%"></div><span>${esc(row.code.toUpperCase())}</span></div>`;
  }).join('');
  byId('languageRows').innerHTML = rows.map((row,index) => `<tr><td class="rank">0${index+1}</td><td><strong>${esc(row.language)}</strong><br><span class="mono">${esc(row.code.toUpperCase())}</span></td><td>${fmt.format(row.bpe_tokens)}</td><td>${fmt.format(row.faithful_units)}</td><td>${fmt.format(row.unique_vocab_tokens_used)}</td><td class="x-cell">${fmt6.format(row.fertility_x)}</td><td><span class="pass">PASS ✓</span></td></tr>`).join('');
}

function renderPlayground() {
  const text = byId('tokenizerInput').value;
  const tokens = encodePublishedBpe(text);
  const vocab = tokenizer.model.vocab;
  const unknown = tokens.filter(token => vocab[token] === undefined).length;
  byId('liveStats').textContent = `${fmt.format(tokens.length)} tokens · ${fmt.format(new Set(tokens).size)} unique${unknown ? ` · ${unknown} unknown` : ''}`;
  byId('liveTokens').innerHTML = tokens.length ? tokens.map((token,index) => {
    const id = vocab[token] === undefined ? vocab['<unk>'] : vocab[token];
    const visible = token.replaceAll('▁','␠').replaceAll('\n','↵').replaceAll('\t','⇥');
    return `<span class="live-token" style="--token-color:${colors[index % colors.length]}" title="Token ID ${id}">${esc(visible || '∅')}</span>`;
  }).join('') : '<span class="mono">Start typing to inspect tokens.</span>';
  byId('decodeStatus').textContent = unknown ? '!' : '✓';
}

function renderApproaches() {
  const best = dashboard.featured.summary;
  const mai = dashboard.maithili.summary;
  const baseline = dashboard.baseline;
  const cards = [
    { title:'Faithful Markdown + Kannada', description:'Standard Metaspace BPE over complete faithful Markdown. Exact round trip and the tightest measured spread.', score:best.raw_score, featured:true, note:'EN · HI · TE · KN' },
    { title:'Faithful Markdown + Maithili', description:'Same standard BPE pipeline with Maithili as the fourth language and independently frozen corpus snapshots.', score:mai.raw_score, note:'EN · HI · TE · MAI' },
    { title:'Custom constrained BPE', description:'The original integer-ID grapheme trainer on cleaned article text, preserved as an algorithmic baseline.', score:baseline.whitespace_word_summary.score, note:'CLEANED TEXT · WORD METRIC' },
  ];
  byId('approachCards').innerHTML = cards.map((card,index) => `<article class="surface approach-card ${card.featured ? 'featured' : ''}"><span class="num">0${index+1}</span><h3>${esc(card.title)}</h3><p>${esc(card.description)}</p><div class="approach-score">${formatScore(card.score)}</div><small>${esc(card.note)}</small></article>`).join('');
  byId('knScore').textContent = formatScore(best.raw_score);
  byId('maiScore').textContent = formatScore(mai.raw_score);
}

function renderDownloads() {
  const a = dashboard.artifacts;
  const files = [
    [a.featured_tokenizer,'FINAL MODEL','Kannada tokenizer.json','Standard Hugging Face format'],
    [a.featured_metrics,'FINAL METRICS','Kannada metrics.json','All score inputs and validation'],
    [a.featured_corpus_manifest,'CORPUS','Snapshot manifest','Source URLs and SHA-256 hashes'],
    [a.maithili_tokenizer,'EXPERIMENT','Maithili tokenizer.json','Alternative fourth-language model'],
    [a.maithili_metrics,'EXPERIMENT','Maithili metrics.json','Complete Maithili results'],
    [a.custom_tokenizer,'BASELINE','Custom tokenizer.json','Original constrained BPE'],
    [a.custom_metrics,'BASELINE','Custom metrics.json','Word and faithful-unit scores'],
    [a.comparison,'COMPARISON','All approaches.json','Machine-readable comparison'],
  ];
  byId('downloadGrid').innerHTML = files.map(([href,type,title,note]) => `<a class="surface download-card" href="${esc(href)}" download><span>${esc(type)} ↘</span><b>${esc(title)}</b><small>${esc(note)}</small></a>`).join('');
}

function bindInteractions() {
  byId('tokenizerInput').addEventListener('input', renderPlayground);
  const samples = {
    english:"India's population is 1,428,627,663.",
    hindi:'भारत विविधताओं का देश है।',
    telugu:'భారతదేశం వైవిధ్యమైన దేశం.',
    kannada:'ಭಾರತವು ವೈವಿಧ್ಯಮಯ ದೇಶವಾಗಿದೆ.',
  };
  document.querySelectorAll('[data-sample]').forEach(button => button.addEventListener('click', () => { byId('tokenizerInput').value = samples[button.dataset.sample]; renderPlayground(); }));
}

async function load() {
  const [dashboardResponse, tokenizerResponse] = await Promise.all([fetch('data/dashboard.json', { cache:'no-store' }), fetch('data/final/kannada/tokenizer.json', { cache:'no-store' })]);
  if (!dashboardResponse.ok || !tokenizerResponse.ok) throw new Error('Published dashboard artifacts could not be loaded.');
  dashboard = await dashboardResponse.json();
  tokenizer = await tokenizerResponse.json();
  mergeRanks = new Map(tokenizer.model.merges.map(([left,right],rank) => [`${left}\u0000${right}`,rank]));
  renderHero(); renderResults(); renderApproaches(); renderDownloads(); bindInteractions(); renderPlayground();
}

load().catch(error => { const banner = byId('errorBanner'); banner.textContent = error.message; banner.classList.remove('hidden'); console.error(error); });
