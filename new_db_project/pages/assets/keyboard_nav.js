// assets/keyboard_nav.js

console.log("keyboard_nav.js loaded");

document.addEventListener('keydown', e => {
  if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;

  // grab all the things we want to navigate
  const focusables = Array.from(
    document.querySelectorAll(
      '.sector-summary, .industry-summary, button.ticker-btn'
    )
  );
  console.log("focusables:", focusables);

  const active = document.activeElement;
  console.log("activeElement:", active);

  const idx = focusables.indexOf(active);
  console.log("active index:", idx);
  if (idx === -1) return;

  const nextIdx = e.key === 'ArrowDown' ? idx + 1 : idx - 1;
  console.log("moving to index:", nextIdx);
  if (nextIdx >= 0 && nextIdx < focusables.length) {
    focusables[nextIdx].focus();
    e.preventDefault();
  }
});