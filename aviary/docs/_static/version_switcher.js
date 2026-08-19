/*
 * Aviary version switcher.
 *
 * Injects two things into every built doc page:
 *   1. A <select> dropdown above the left sidebar TOC listing every version
 *      published on gh-pages. On change, it navigates to the same relative
 *      page path under the chosen version, falling back to that version's
 *      root if the same-path page doesn't exist.
 *   2. A yellow banner at the top of the content area when the current
 *      version is not the latest stable release.
 *
 * The site layout it assumes (produced by publish_docs.py):
 *   /Aviary/versions.json   -- machine-readable version list
 *   /Aviary/dev/            -- development build
 *   /Aviary/latest/         -- alias for highest stable release
 *   /Aviary/vX.Y.Z/         -- one dir per release tag
 *
 * Runs identically in production and on `python -m http.server` previews:
 * the base path is derived from `location.pathname`, not hard-coded.
 */
(function () {
  'use strict';

  // ---- Locate site root + current version from the URL ------------------
  //
  // A doc page URL looks like:
  //   https://openmdao.github.io/Aviary/dev/getting_started.html
  //                             ^^^^^^^ ^^^ ^^^^^^^^^^^^^^^^^^^
  //                             base    ver page path
  //
  // The version segment is whatever immediately follows the site base. We
  // don't know the base statically (fork vs upstream), so we identify the
  // version segment by matching against the known shape: 'dev', 'latest',
  // or 'vX...'. Everything before it is the base; everything after is the
  // in-version page path.
  function parseLocation() {
    const parts = location.pathname.split('/').filter(Boolean);
    let versionIdx = -1;
    for (let i = 0; i < parts.length; i++) {
      const p = parts[i];
      if (p === 'dev' || p === 'latest' || /^v\d/.test(p)) {
        versionIdx = i;
        break;
      }
    }
    if (versionIdx < 0) return null;   // not on a versioned page
    return {
      base: '/' + parts.slice(0, versionIdx).join('/') + '/',
      version: parts[versionIdx],
      pagePath: parts.slice(versionIdx + 1).join('/'),
    };
  }

  const loc = parseLocation();
  if (!loc) return;

  // ---- Fetch versions.json ---------------------------------------------
  fetch(loc.base + 'versions.json', {cache: 'no-store'})
    .then(function (r) { return r.ok ? r.json() : null; })
    .then(function (data) {
      if (!data || !Array.isArray(data.versions)) return;
      renderSwitcher(loc, data);
      renderBanner(loc, data);
    })
    .catch(function () { /* silent: no versions.json => no switcher */ });

  // ---- Sidebar dropdown -------------------------------------------------
  function renderSwitcher(loc, data) {
    // Insert the switcher right after the logo inside .sidebar-primary-item.
    // Final order in the sidebar: logo -> switcher -> TOC.
    const logo = document.querySelector('.sidebar-primary-item .navbar-brand');
    if (!logo) return;

    const wrap = document.createElement('div');
    wrap.className = 'aviary-version-switcher';

    const label = document.createElement('label');
    // The for attribute binds the label to the form control whose id matches the value.
    label.setAttribute('for', 'aviary-version-select');
    label.textContent = 'Version';
    wrap.appendChild(label);

    const select = document.createElement('select');
    select.id = 'aviary-version-select';

    // Populate: dev first, then releases in the order versions.json gave
    // us (publish_docs.py already sorts newest-first).
    data.versions.forEach(function (v) {
      const opt = document.createElement('option');
      opt.value = v.name;
      let text = v.name;
      if (v.kind === 'dev') text += ' (development)';
      else if (data.latest && v.name === data.latest) text += ' (latest stable)';
      opt.textContent = text;
      if (v.name === loc.version) opt.selected = true;
      select.appendChild(opt);
    });

    // Special case: current URL uses /latest/ alias. The dropdown shows the
    // resolved version name as selected; add a synthetic 'latest' option so
    // the user can see they're on the alias.
    if (loc.version === 'latest') {
      const opt = document.createElement('option');
      opt.value = 'latest';
      opt.textContent = 'latest' + (data.latest ? ' → ' + data.latest : '');
      opt.selected = true;
      select.insertBefore(opt, select.firstChild);
    }

    select.addEventListener('change', function () {
      navigateToVersion(loc, select.value);
    });

    wrap.appendChild(select);
    logo.insertAdjacentElement('afterend', wrap);
  }

  // ---- Yellow banner on non-latest pages --------------------------------
  function renderBanner(loc, data) {
    if (!data.latest) return;                       // no stable release yet
    if (loc.version === 'latest') return;           // already on alias
    if (loc.version === data.latest) return;        // on the resolved latest

    let message;
    if (loc.version === 'dev') {
      message = 'You are viewing the development docs. Latest stable release is ' +
                anchor(loc, data.latest, data.latest) + '.';
    } else {
      message = 'You are viewing docs for ' + loc.version +
                '. Latest stable release is ' +
                anchor(loc, data.latest, data.latest) + '.';
    }

    const main = document.querySelector('.bd-main');
    if (!main) return;

    const banner = document.createElement('div');
    banner.className = 'aviary-version-banner';
    banner.innerHTML = message;
    main.prepend(banner);
  }

  // Build an <a> string that jumps to `targetVersion` at the same page path,
  // for embedding in the banner text via innerHTML. (No user-controlled
  // strings involved, so innerHTML is safe here.)
  function anchor(loc, targetVersion, label) {
    const href = loc.base + targetVersion + '/' + loc.pagePath;
    return '<a href="' + href + '">' + label + '</a>';
  }

  // ---- Cross-version navigation with same-path fallback -----------------
  //
  // Try the equivalent page under the target version. If it 404s, fall
  // back to that version's root (which always exists as index.html).
  function navigateToVersion(loc, targetVersion) {
    if (targetVersion === loc.version) return;
    const samePath = loc.base + targetVersion + '/' + loc.pagePath;
    const rootPath = loc.base + targetVersion + '/';

    if (!loc.pagePath) {
      location.href = rootPath;
      return;
    }

    // HEAD probe. If it fails (network error or non-2xx) we go to the root.
    fetch(samePath, {method: 'HEAD'})
      .then(function (r) {
        location.href = r.ok ? samePath : rootPath;
      })
      .catch(function () {
        location.href = rootPath;
      });
  }
})();
