console.log("📘 keyboard_nav.js ✅");

document.addEventListener('keydown', (e) => {
  // debug log every key
  console.log("key:", e.key, "code:", e.code, "shift:", e.shiftKey);

  // only when Shift is held
  if (!e.shiftKey) return;

  // pick your key: ArrowLeft/ArrowRight or '<'/'>'
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;

  const links = Array.from(
  document.querySelectorAll('a[href^="/"]')
);
  console.log("found links:", links.map(l => l.href));

  const activeIndex = links.findIndex(link =>
    link.classList.contains('active') ||
    window.location.pathname === new URL(link.href).pathname
  );
  console.log("activeIndex:", activeIndex);
  if (activeIndex < 0) return;

  let idx = (e.key === 'ArrowLeft') ? activeIndex - 1 : activeIndex + 1;
  if (idx < 0 || idx >= links.length) return;

  console.log("navigating to:", links[idx].href);
  window.location.href = links[idx].href;
  e.preventDefault();
});