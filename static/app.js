const API_ROOT = '/';

// Cached state for diffing
let _svg = null;
let _prevPlaceIds = new Set();

async function fetchState(){
  const res = await fetch(API_ROOT + 'state');
  return res.json();
}

// ── SVG bootstrap (called once) ────────────────────────────────────────────

function createSvg(width, height){
  const svgNS = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(svgNS, 'svg');
  svg.setAttribute('width', width);
  svg.setAttribute('height', height);
  svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
  svg.style.border = '2px solid #333';
  svg.style.background = '#e8f4e8';
  return svg;
}

// ── Place renderers (static geometry, rendered once) ───────────────────────

function buildPlaceElement(svg, place){
  let element;
  if (place.polygon) {
    element = document.createElementNS(svg.namespaceURI, 'polygon');
    element.setAttribute('points', place.polygon.map(p => `${p.x},${p.y}`).join(' '));
    element.setAttribute('fill',         place.render.fillColor);
    element.setAttribute('stroke',       place.render.strokeColor);
    element.setAttribute('stroke-width', place.render.strokeWidth);
    if (place.render.opacity !== 1.0)
      element.setAttribute('opacity', place.render.opacity);
  } else if (place.points) {
    element = document.createElementNS(svg.namespaceURI, 'polyline');
    element.setAttribute('points', place.points.map(p => `${p.x},${p.y}`).join(' '));
    element.setAttribute('stroke',       place.render.strokeColor);
    element.setAttribute('stroke-width', place.render.strokeWidth);
    element.setAttribute('stroke-linecap',  'round');
    element.setAttribute('stroke-linejoin', 'round');
    element.setAttribute('fill', 'none');
    if (place.render.opacity !== 1.0)
      element.setAttribute('opacity', place.render.opacity);
  }
  if (!element) return null;

  element.id = `place-${place.id}`;
  if (place.type === 'building') {
    element.style.cursor = 'pointer';
    element.addEventListener('click', () => showPlace(place));
  }
  return element;
}

// ── Entity renderer ────────────────────────────────────────────────────────

function buildEntityGroup(svg, entity){
  const r = entity.render || {color: '#ff6b35', strokeColor: '#000', radius: 6, strokeWidth: 1.5};
  const g = document.createElementNS(svg.namespaceURI, 'g');
  g.id = `entity-${entity.id}`;

  const circ = document.createElementNS(svg.namespaceURI, 'circle');
  circ.setAttribute('cx', entity.pos.x);
  circ.setAttribute('cy', entity.pos.y);
  circ.setAttribute('r',  r.radius);
  circ.setAttribute('fill',         r.color);
  circ.setAttribute('stroke',       r.strokeColor);
  circ.setAttribute('stroke-width', r.strokeWidth);
  circ.style.cursor = 'pointer';
  circ.addEventListener('click', () => showEntity(entity));
  g.appendChild(circ);

  const text = document.createElementNS(svg.namespaceURI, 'text');
  text.setAttribute('x', entity.pos.x);
  text.setAttribute('y', entity.pos.y + 15);
  text.setAttribute('text-anchor', 'middle');
  text.setAttribute('font-size', '9');
  text.setAttribute('fill', '#000');
  text.textContent = entity.name;
  g.appendChild(text);

  return g;
}

// ── Full initial draw ──────────────────────────────────────────────────────

function initDraw(state){
  const container = document.getElementById('map-container');
  container.innerHTML = '';
  _svg = createSvg(state.width, state.height);
  _prevPlaceIds = new Set();

  // Places – sorted by zIndex, rendered once
  const sortedPlaces = [...state.places].sort((a, b) => a.render.zIndex - b.render.zIndex);
  for (const place of sortedPlaces) {
    const el = buildPlaceElement(_svg, place);
    if (el) {
      _svg.appendChild(el);
      _prevPlaceIds.add(place.id);
    }
  }

  // Entities – sorted by zIndex
  const sortedEntities = [...state.entities].sort((a, b) => (a.render?.zIndex||100) - (b.render?.zIndex||100));
  for (const entity of sortedEntities) {
    _svg.appendChild(buildEntityGroup(_svg, entity));
  }

  container.appendChild(_svg);
  document.getElementById('tick-count').textContent = 'tick: ' + state.tick;
}

// ── Incremental update (called every tick) ─────────────────────────────────

function updateDraw(state){
  if (!_svg) { initDraw(state); return; }

  // ── Places: add new ones, leave existing untouched ──
  const sortedPlaces = [...state.places].sort((a, b) => a.render.zIndex - b.render.zIndex);
  for (const place of sortedPlaces) {
    if (!_prevPlaceIds.has(place.id)) {
      // Find the first entity group so we insert places below entities
      const firstEntity = _svg.querySelector('g[id^="entity-"]');
      const el = buildPlaceElement(_svg, place);
      if (el) {
        _svg.insertBefore(el, firstEntity || null);
        _prevPlaceIds.add(place.id);
      }
    }
  }

  // ── Entities: update position/text or create if new ──
  for (const entity of state.entities) {
    const g = _svg.querySelector(`#entity-${entity.id}`);
    if (g) {
      // Update circle position
      const circ = g.querySelector('circle');
      circ.setAttribute('cx', entity.pos.x);
      circ.setAttribute('cy', entity.pos.y);
      // Update label position
      const text = g.querySelector('text');
      text.setAttribute('x', entity.pos.x);
      text.setAttribute('y', entity.pos.y + 15);
      // Refresh click data (wealth/happiness may have changed)
      circ.onclick = () => showEntity(entity);
    } else {
      // New entity appeared mid-simulation
      _svg.appendChild(buildEntityGroup(_svg, entity));
    }
  }

  // ── Remove entities that are no longer in state ──
  const liveIds = new Set(state.entities.map(e => e.id));
  _svg.querySelectorAll('g[id^="entity-"]').forEach(g => {
    const id = parseInt(g.id.replace('entity-', ''));
    if (!liveIds.has(id)) g.remove();
  });

  document.getElementById('tick-count').textContent = 'tick: ' + state.tick;
}

// ── Info panels ────────────────────────────────────────────────────────────

function showEntity(e){
  document.getElementById('info').innerHTML = `
    <h3>${e.name}</h3>
    <p><strong>ID:</strong> ${e.id}</p>
    <p><strong>Position:</strong> (${e.pos.x.toFixed(1)}, ${e.pos.y.toFixed(1)})</p>
    <p><strong>Wealth:</strong> ${e.wealth.toFixed(2)}</p>
    <p><strong>Happiness:</strong> ${e.happiness.toFixed(2)}</p>
  `;
}

function showPlace(place){
  let attributesHtml = '';
  for (const [key, value] of Object.entries(place.attributes))
    attributesHtml += `<p><strong>${key}:</strong> ${value}</p>`;
  document.getElementById('info').innerHTML = `
    <h3>${place.name}</h3>
    <p><strong>Type:</strong> ${place.type}</p>
    <p><strong>ID:</strong> ${place.id}</p>
    ${attributesHtml}
  `;
}

// ── Controls ───────────────────────────────────────────────────────────────

document.getElementById('tick').addEventListener('click', async () => {
  await fetch(API_ROOT + 'tick', {method: 'POST'});
  const s = await fetchState();
  updateDraw(s);
});

// Initial load – full draw
fetchState().then(initDraw).catch(err => console.error(err));
