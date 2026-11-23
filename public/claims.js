/* Research Assistant: claim badges.
   The answer carries one badge per checked sentence, a Chainlit element link named
   "<glyph> C<n>" (glyph per status, see app/render.py CLAIM_GLYPHS). Element links have no
   status hook, so this script adds the CSS classes "claim-badge claim-<status>" (styled in
   public/custom.css). Without it the badges still show their glyph. */
(function () {
  "use strict";
  var STATUS = { "✓": "supported", "◐": "partial", "✗": "unsupported", "?": "uncited" };
  var TITLE = {
    supported: "Supported by the cited source",
    partial: "Partially supported by the cited source",
    unsupported: "Not supported by the cited source",
    uncited: "States a fact without a citation"
  };
  var NAME = /^(✓|◐|✗|\?) C\d+$/;

  function tag(root) {
    var links = root.querySelectorAll ? root.querySelectorAll("a.element-link:not(.claim-badge)") : [];
    for (var i = 0; i < links.length; i++) {
      var a = links[i];
      var m = NAME.exec((a.textContent || "").trim());
      if (!m) continue;
      var status = STATUS[m[1]];
      a.classList.add("claim-badge", "claim-" + status);
      a.setAttribute("title", TITLE[status]);
      a.setAttribute("aria-label", TITLE[status] + ", claim " + a.textContent.trim().split(" C")[1]);
    }
  }

  function start() {
    tag(document);
    new MutationObserver(function (records) {
      for (var i = 0; i < records.length; i++) {
        var added = records[i].addedNodes;
        for (var j = 0; j < added.length; j++) {
          if (added[j].nodeType === 1) tag(added[j].parentNode || added[j]);
        }
      }
    }).observe(document.body, { childList: true, subtree: true });
  }

  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", start);
  else start();
})();
