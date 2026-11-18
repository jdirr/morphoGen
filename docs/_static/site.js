// What the pages of the site do beyond their theme, without any library: a button that
// copies each code block, a filter of the cards of the gallery, and the screenshots of
// the tutorials enlarged by a click. Each part looks for its elements and does nothing
// on a page without them.
(function () {
  'use strict';

  // --- copy a code block ------------------------------------------------------------

  function copyText(text) {
    if (navigator.clipboard && window.isSecureContext) {
      return navigator.clipboard.writeText(text);
    }
    return new Promise(function (resolve, reject) {
      // a page opened from a file, or an old browser: a hidden text area and the command
      var area = document.createElement('textarea');
      area.value = text;
      area.setAttribute('readonly', '');
      area.style.position = 'fixed';
      area.style.opacity = '0';
      document.body.appendChild(area);
      area.select();
      var done = false;
      try {
        done = document.execCommand('copy');
      } catch (error) {
        done = false;
      }
      document.body.removeChild(area);
      if (done) resolve();
      else reject(new Error('copy refused'));
    });
  }

  function copyButtons() {
    document.querySelectorAll('div.highlight').forEach(function (block) {
      var pre = block.querySelector('pre');
      if (!pre || block.querySelector('button.copy-code')) return;
      var button = document.createElement('button');
      button.type = 'button';
      button.className = 'copy-code';
      button.textContent = 'Copy';
      button.setAttribute('aria-label', 'Copy this code');
      button.addEventListener('click', function () {
        var said = function (word) {
          button.textContent = word;
          button.classList.add('copied');
          window.setTimeout(function () {
            button.textContent = 'Copy';
            button.classList.remove('copied');
          }, 1600);
        };
        copyText(pre.textContent.replace(/\n$/, '')).then(
          function () { said('Copied'); },
          function () { said('Select it'); }
        );
      });
      block.appendChild(button);
    });
  }

  // --- filter the cards of the gallery ----------------------------------------------

  function galleryFilter() {
    var cards = Array.prototype.slice.call(document.querySelectorAll('.gallery-card[id^="gallery-"]'));
    if (cards.length < 12) return; // the gallery of examples, not a few cards of a page
    var first = cards[0].closest('section');
    if (!first || !first.parentNode) return;
    var sections = [];
    cards.forEach(function (card) {
      var section = card.closest('section');
      if (section && sections.indexOf(section) < 0) sections.push(section);
    });
    var texts = cards.map(function (card) {
      var details = card.querySelector('details'); // the configuration stays out of the search
      var text = card.textContent;
      if (details) text = text.replace(details.textContent, ' ');
      var heading = card.closest('section').querySelector('h2');
      return ((heading ? heading.textContent : '') + ' ' + text).toLowerCase();
    });

    var box = document.createElement('div');
    box.className = 'gallery-filter';
    var input = document.createElement('input');
    input.type = 'search';
    input.placeholder = 'Filter the ' + cards.length + ' examples: a model, a solver, a word...';
    input.setAttribute('aria-label', 'Filter the examples of the gallery');
    var count = document.createElement('span');
    count.setAttribute('aria-live', 'polite');
    box.appendChild(input);
    box.appendChild(count);
    first.parentNode.insertBefore(box, first);

    function apply() {
      var words = input.value.toLowerCase().split(/\s+/).filter(Boolean);
      var shown = 0;
      cards.forEach(function (card, k) {
        var keep = words.every(function (word) { return texts[k].indexOf(word) >= 0; });
        card.classList.toggle('filtered', !keep);
        if (keep) shown += 1;
      });
      sections.forEach(function (section) {
        var any = section.querySelector('.gallery-card[id^="gallery-"]:not(.filtered)');
        section.classList.toggle('filtered', !any);
      });
      count.textContent = words.length ? shown + ' of ' + cards.length : '';
    }

    input.addEventListener('input', apply);
    var asked = new URLSearchParams(window.location.search).get('filter');
    if (asked) {
      input.value = asked; // gallery.html?filter=fracture
      apply();
    }
  }

  // --- enlarge a screenshot ---------------------------------------------------------

  function lightbox() {
    var shots = document.querySelectorAll('img.tutorial-screenshot');
    if (!shots.length) return;
    var open = null;
    function close() {
      if (open) {
        open.remove();
        open = null;
      }
    }
    shots.forEach(function (shot) {
      shot.setAttribute('tabindex', '0');
      shot.setAttribute('role', 'button');
      shot.setAttribute('title', 'Click to enlarge');
      function show() {
        close();
        open = document.createElement('div');
        open.className = 'lightbox';
        var large = document.createElement('img');
        large.src = shot.currentSrc || shot.src;
        large.alt = shot.alt;
        open.appendChild(large);
        open.addEventListener('click', close);
        document.body.appendChild(open);
      }
      shot.addEventListener('click', show);
      shot.addEventListener('keydown', function (event) {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          show();
        }
      });
    });
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') close();
    });
  }

  function start() {
    copyButtons();
    galleryFilter();
    lightbox();
  }

  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', start);
  else start();
})();
