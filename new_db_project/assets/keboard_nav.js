// assets/keyboard_nav.js

document.addEventListener('keydown', (e) => {
  // only when Shift is held
  if (!e.shiftKey) return;

  // look for all the nav links (e.g. from dbc.NavLink)
  const links = Array.from(document.querySelectorAll('.nav-link'))
                     .filter(el => el.href);

  if (!links.length) return;

  // find which one is currently active
  const activeIndex = links.findIndex(el =>
    el.classList.contains('active') ||
    window.location.pathname === new URL(el.href).pathname
  );
  if (activeIndex === -1) return;

  let targetIndex = null;
  if (e.key === 'ArrowLeft') {
    targetIndex = activeIndex - 1;
  } else if (e.key === 'ArrowRight') {
    targetIndex = activeIndex + 1;
  } else {
    return;
  }

  // if in range, navigate
  if (targetIndex >= 0 && targetIndex < links.length) {
    window.location.href = links[targetIndex].href;
    e.preventDefault();
  }
});
