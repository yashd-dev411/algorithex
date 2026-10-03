// Algorithex theme bootstrap: brand the favicon (additive, no app logic touched).
(function () {
  function brandFavicon() {
    try {
      var link = document.querySelector('link[rel="icon"]');
      if (!link) {
        link = document.createElement('link');
        link.rel = 'icon';
        document.head.appendChild(link);
      }
      link.type = 'image/svg+xml';
      link.href = '/logo.svg?v=ax1';
    } catch (e) { /* noop */ }
  }
  brandFavicon();
})();