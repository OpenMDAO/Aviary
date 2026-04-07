/*
 * collapsible_headers.js
 *
 * Automatically makes every h2-h6 section in a Jupyter Book page
 * collapsible without requiring any per-header markup from authors.
 *
 * How it works:
 *   1. On DOMContentLoaded, init() walks every <section> element inside
 *      the main article container.
 *   2. For each section that has a direct h2-h6 child, the DOM is
 *      restructured: all children are moved into a <details> block whose
 *      <summary> contains the heading.  The section starts open.
 *   3. If the page URL contains a fragment (#anchor), openParentsForHash()
 *      forces any <details> ancestor of that target element to be open so
 *      the linked content is always visible on arrival.
 *
 * Styling is handled entirely in custom.css (aviary-collapsible-* classes).
 * No external dependencies are required.
 */
(function () {
  "use strict";

  /*
   * getHeadingLevel
   *
   * Returns the numeric heading level (1-6) for a given element, or null
   * if the element is not a heading tag.  Used to guard against accidentally
   * processing h1 page-title headings.
   *
   * @param {Element} heading - A DOM element to inspect.
   * @returns {number|null}
   */
  function getHeadingLevel(heading) {
    if (!heading || !heading.tagName) {
      return null;
    }
    var match = heading.tagName.match(/^H([1-6])$/);
    return match ? parseInt(match[1], 10) : null;
  }

  /*
   * hasHeadingContent
   *
   * Returns true if the section element contains at least one child element
   * other than the heading itself.  Sections that only contain a heading
   * (no body content) are left untouched because wrapping them in a
   * collapsible <details> would produce an empty, misleading toggle.
   *
   * @param {Element} section - The <section> element to check.
   * @param {Element} heading - The heading element within that section.
   * @returns {boolean}
   */
  function hasHeadingContent(section, heading) {
    if (!section) {
      return false;
    }

    var child = section.firstElementChild;
    while (child) {
      if (child !== heading) {
        return true;
      }
      child = child.nextElementSibling;
    }
    return false;
  }

  /*
   * buildCollapsibleSection
   *
   * Restructures a single <section> into a collapsible <details> block:
   *
   *   Before:                         After:
   *   <section>                       <section>
   *     <h2>Title</h2>                  <details open>
   *     <p>Body...</p>                    <summary><h2>Title</h2></summary>
   *   </section>                          <div class="...-body">
   *                                         <p>Body...</p>
   *                                       </div>
   *                                     </details>
   *                                   </section>
   *
   * All existing children (including the heading) are moved, preserving
   * their event listeners and sub-tree structure.  The section starts open
   * so the page layout is unchanged on first load.
   *
   * @param {Element} section - The <section> element to transform.
   * @param {Element} heading - The direct h2-h6 child to use as the summary.
   */
  function buildCollapsibleSection(section, heading) {
    if (!section || !heading) {
      return;
    }

    if (!hasHeadingContent(section, heading)) {
      return;
    }

    var details = document.createElement("details");
    details.className = "aviary-collapsible-section";
    details.open = true; /* start expanded so the page looks unchanged by default */

    var summary = document.createElement("summary");
    summary.className = "aviary-collapsible-summary";
    summary.appendChild(heading);

    var body = document.createElement("div");
    body.className = "aviary-collapsible-body";

    /* Drain all remaining children of the section into the body div.
     * We loop on firstChild (not firstElementChild) to also capture
     * text nodes and comments that may sit between elements. */
    while (section.firstChild) {
      body.appendChild(section.firstChild);
    }

    details.appendChild(summary);
    details.appendChild(body);
    section.appendChild(details);
  }

  /*
   * openParentsForHash
   *
   * When a page is loaded with a URL fragment (e.g. page.html#my-section),
   * the target element may be inside a collapsed <details> block, making its
   * content invisible.  This function ensures the anchor destination is always
   * fully visible in two steps:
   *
   *   Step 1 — Open the section the anchor belongs to.
   *     In Sphinx/JupyterBook the anchor id is placed on the <section> element
   *     itself, not on the heading.  After our JS transformation the <details>
   *     block is a direct child of that <section>, so we look for it with
   *     :scope > details and open it so the section body is revealed.
   *
   *   Step 2 — Open all ancestor <details> blocks.
   *     If the target section is nested inside another collapsed section we
   *     walk up the DOM and force every <details> ancestor open too, ensuring
   *     the full path from the page root to the destination is visible.
   *
   * Also registered as a "hashchange" listener so in-page navigation
   * (e.g. clicking a TOC link) triggers the same behaviour.
   */
  function openParentsForHash() {
    if (!window.location.hash) {
      return;
    }

    var target = document.getElementById(window.location.hash.slice(1));
    if (!target) {
      return;
    }

    /* Step 1: if the anchor points at a <section> whose content was wrapped
     * in a <details> child by buildCollapsibleSection, open that details so
     * the body of the destination section is expanded and visible. */
    var ownDetails = target.querySelector(":scope > details.aviary-collapsible-section");
    if (ownDetails) {
      ownDetails.open = true;
    }

    /* Step 2: walk up the DOM and open every enclosing <details> block so
     * that any parent sections hiding this target are also expanded. */
    var node = target;
    while (node) {
      if (node.tagName === "DETAILS") {
        node.open = true;
      }
      node = node.parentElement;
    }
  }

  /*
   * init
   *
   * Entry point.  Queries the article element that Jupyter Book / sphinx-book-
   * theme places all page content inside, then iterates every <section>
   * descendant.  Sections are processed deepest-first because
   * querySelectorAll returns elements in document order (parent before
   * child), but buildCollapsibleSection moves children into a new subtree,
   * so inner sections are handled before their parent drains them.
   *
   * A data attribute (data-aviary-collapsible-processed) is set on each
   * section after processing to ensure idempotency in case init is called
   * more than once.
   */
  function init() {
    var article = document.querySelector("main.bd-main article.bd-article");
    if (!article) {
      return;
    }

    var sections = article.querySelectorAll("section");
    sections.forEach(function (section) {
      /* Skip sections that were already transformed. */
      if (section.dataset.aviaryCollapsibleProcessed === "true") {
        return;
      }

      /* Only collapse sections whose immediate heading is h2 or deeper;
       * h1 is the page title and should never be collapsible. */
      var heading = section.querySelector(":scope > h2, :scope > h3, :scope > h4, :scope > h5, :scope > h6");
      if (!heading) {
        return;
      }

      var level = getHeadingLevel(heading);
      if (level === null || level < 2) {
        return;
      }

      section.dataset.aviaryCollapsibleProcessed = "true";
      buildCollapsibleSection(section, heading);
    });

    openParentsForHash();
    window.addEventListener("hashchange", openParentsForHash);
  }

  /* Run init as soon as the DOM is ready.  If the script is deferred or
   * placed at the end of <body> the document may already be interactive,
   * so we handle both cases. */
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
