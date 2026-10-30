"""
Publish Aviary documentation to the gh-pages branch under a versioned subdirectory.

Layout produced on gh-pages:
    /                  landing page (index.html) with a version picker
    /versions.json     machine-readable list consumed by the landing page
    /dev/              build from tip of main
    /latest/           mirror of the highest stable release
    /vX.Y.Z/           one directory per release tag

Usage:
    # Publish the current HTML build as the "dev" version
    python publish_docs.py --kind dev --html-dir _build/html

    # Publish as a release tag
    python publish_docs.py --kind tag --tag v1.0.1 --html-dir _build/html

    # Local preview without touching git (writes a full site tree to a directory)
    python publish_docs.py --kind dev --html-dir _build/html \
                           --out-dir /tmp/ghp-preview --dry-run

Design notes:
    - Modeled on OpenMDAO's upload_doc_version.py (same packaging.version-based
      tag ordering, same "release -> versioned subdir, else -> latest/dev" split),
      but writes into a local git worktree of gh-pages rather than rsyncing to a
      remote host. That way each publish preserves every other subdirectory that
      already exists on gh-pages.
    - "Stable" release = tag whose PEP 440 version has no pre/dev/local component
      (no -rc, -a, -b, -dev, +local). Only stable releases become /latest/.
"""

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from packaging.version import InvalidVersion, Version

SAFE_SUBDIR_RE = re.compile(r'^[A-Za-z0-9._-]+$')


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------


def _run(cmd, cwd=None, check=True, capture=False):
    """Run a subprocess command, echoing it for CI logs."""
    print(f'$ {" ".join(cmd)}' + (f'   (cwd={cwd})' if cwd else ''))
    if capture:
        result = subprocess.run(cmd, cwd=cwd, check=check, capture_output=True, text=True)
        return result.stdout.strip()
    subprocess.run(cmd, cwd=cwd, check=check)
    return None


def get_release_tags(repo_root, extra_tags=()):
    """
    Return release tags sorted newest-first, filtered to those parseable as
    PEP 440 versions.

    `extra_tags` are additional tag names to include even if `git tag -l` did
    not surface them (e.g. the `--tag` argument the user just asked us to
    publish, which is always relevant regardless of what git thinks).
    """
    out = _run(['git', 'tag', '-l', 'v*.*.*'], cwd=repo_root, capture=True)
    raw = {t for t in out.split() if t} | set(extra_tags)
    print(f'  discovered {len(raw)} candidate tag(s): {sorted(raw)}')

    parsed = []
    for t in raw:
        # Strip leading 'v' for Version parsing; keep the original tag string.
        try:
            v = Version(t.lstrip('v'))
        except InvalidVersion:
            print(f'  skipping unparseable tag: {t}')
            continue
        parsed.append((v, t))

    parsed.sort(key=lambda p: p[0], reverse=True)
    return parsed  # list of (Version, "vX.Y.Z")


def pick_latest_stable(parsed_tags):
    """Return the tag string of the highest stable release, or None."""
    for v, tag in parsed_tags:
        if v.is_prerelease or v.is_devrelease or v.local:
            continue
        return tag
    return None


# ---------------------------------------------------------------------------
# versions.json + landing page
# ---------------------------------------------------------------------------


def build_versions_json(existing_subdirs, parsed_tags, latest_stable):
    """
    Build the versions.json payload.

    `existing_subdirs` is the set of directory names present at the root of the
    site (either the gh-pages worktree, or the --out-dir tree). We only list a
    version in versions.json if its directory actually exists — that way the
    landing page never links to a 404.
    """
    versions = []
    if 'dev' in existing_subdirs:
        versions.append({'name': 'dev', 'path': 'dev/', 'kind': 'dev'})

    for v, tag in parsed_tags:
        if tag not in existing_subdirs:
            continue
        stable = not (v.is_prerelease or v.is_devrelease or v.local)
        versions.append(
            {
                'name': tag,
                'path': f'{tag}/',
                'kind': 'release',
                'stable': stable,
            }
        )

    return {
        'latest': latest_stable if latest_stable in existing_subdirs else None,
        'versions': versions,
    }


LANDING_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Aviary Documentation</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
  :root { color-scheme: light dark; }
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                     "Helvetica Neue", Arial, sans-serif;
         max-width: 720px; margin: 3rem auto; padding: 0 1.25rem;
         line-height: 1.55; }
  h1 { margin-bottom: 0.25rem; }
  .sub { color: #666; margin-top: 0; }
  .primary { display: flex; gap: 0.75rem; flex-wrap: wrap; margin: 1.5rem 0; }
  .primary a { display: inline-block; padding: 0.6rem 1rem; border-radius: 6px;
               text-decoration: none; border: 1px solid #ccc; }
  .primary a.stable { background: #0b5cff; color: white; border-color: #0b5cff; }
  .picker { margin: 2rem 0 1rem; }
  .picker label { display: block; font-weight: 600; margin-bottom: 0.35rem; }
  .picker select { padding: 0.5rem; min-width: 16rem; font-size: 1rem; }
  ul.versions { padding-left: 1.25rem; }
  ul.versions li { margin: 0.15rem 0; }
  .kind-dev { color: #8a5a00; }
  .empty { color: #999; font-style: italic; }
  @media (prefers-color-scheme: dark) {
    body { background: #14161a; color: #e6e6e6; }
    .sub { color: #9aa0a6; }
    .primary a { border-color: #3a3f47; color: #e6e6e6; }
    .primary a.stable { background: #4c8bf5; border-color: #4c8bf5; color: white; }
  }
</style>
</head>
<body>
<h1>Aviary Documentation</h1>
<p class="sub">Pick a version below.</p>

<div class="primary" id="primary-links"></div>

<div class="picker">
  <label for="version-select">All versions</label>
  <select id="version-select"></select>
</div>

<h3>All published versions</h3>
<ul class="versions" id="version-list"></ul>

<script>
(async function () {
  const primary = document.getElementById('primary-links');
  const select = document.getElementById('version-select');
  const list = document.getElementById('version-list');

  let data;
  try {
    const resp = await fetch('versions.json', {cache: 'no-store'});
    data = await resp.json();
  } catch (e) {
    primary.innerHTML = '<span class="empty">Could not load versions.json.</span>';
    return;
  }

  const versions = data.versions || [];
  if (!versions.length) {
    primary.innerHTML = '<span class="empty">No versions published yet.</span>';
    return;
  }

  // Primary buttons: latest stable, then dev.
  if (data.latest) {
    const a = document.createElement('a');
    a.href = 'latest/';
    a.className = 'stable';
    a.textContent = 'Latest stable (' + data.latest + ')';
    primary.appendChild(a);
  }
  const dev = versions.find(v => v.kind === 'dev');
  if (dev) {
    const a = document.createElement('a');
    a.href = dev.path;
    a.textContent = 'Development (dev)';
    primary.appendChild(a);
  }

  // Dropdown + list of every version.
  for (const v of versions) {
    const opt = document.createElement('option');
    opt.value = v.path;
    opt.textContent = v.name + (v.kind === 'dev' ? ' (development)' : '');
    select.appendChild(opt);

    const li = document.createElement('li');
    const a = document.createElement('a');
    a.href = v.path;
    a.textContent = v.name;
    if (v.kind === 'dev') a.className = 'kind-dev';
    li.appendChild(a);
    if (data.latest && v.name === data.latest) {
      li.appendChild(document.createTextNode('  — latest stable'));
    }
    list.appendChild(li);
  }
  select.addEventListener('change', () => {
    if (select.value) window.location.href = select.value;
  });
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Filesystem helpers
# ---------------------------------------------------------------------------


def _validate_subdir(name):
    if not SAFE_SUBDIR_RE.match(name):
        raise ValueError(f'Refusing to use unsafe subdir name: {name!r}')


def _copy_html_into(root, subdir, html_dir):
    """Replace root/subdir with a fresh copy of html_dir."""
    _validate_subdir(subdir)
    dest = root / subdir
    if dest.exists():
        print(f'  removing existing {dest}')
        shutil.rmtree(dest)
    print(f'  copying {html_dir} -> {dest}')
    shutil.copytree(html_dir, dest)


def _write_site_metadata(root, parsed_tags, latest_stable):
    """Write versions.json, index.html, .nojekyll based on what's on disk."""
    existing = {p.name for p in root.iterdir() if p.is_dir()}
    payload = build_versions_json(existing, parsed_tags, latest_stable)

    versions_path = root / 'versions.json'
    versions_path.write_text(json.dumps(payload, indent=2) + '\n')
    print(
        f'  wrote {versions_path}  (latest={payload["latest"]}, count={len(payload["versions"])})'
    )

    (root / 'index.html').write_text(LANDING_HTML)
    print(f'  wrote {root / "index.html"}')

    nojekyll = root / '.nojekyll'
    if not nojekyll.exists():
        nojekyll.write_text('')
        print(f'  wrote {nojekyll}')


# ---------------------------------------------------------------------------
# Publish modes
# ---------------------------------------------------------------------------


def publish_to_worktree(
    repo_root, html_dir, subdir, is_latest, parsed_tags, latest_stable, dry_run, commit_message
):
    """
    Check out gh-pages into a temp worktree, drop in the new subdir, refresh
    metadata, commit, push.
    """
    with tempfile.TemporaryDirectory(prefix='ghp-worktree-') as tmp:
        worktree = Path(tmp) / 'gh-pages'

        # gh-pages might not exist yet on a fresh fork. Try to fetch; if the
        # branch is missing, create an orphan worktree.
        try:
            _run(['git', 'fetch', 'origin', 'gh-pages'], cwd=repo_root)
            _run(['git', 'worktree', 'add', str(worktree), 'origin/gh-pages'], cwd=repo_root)
            _run(['git', 'checkout', '-B', 'gh-pages'], cwd=worktree)
        except subprocess.CalledProcessError:
            print('  gh-pages branch not found; creating a fresh orphan branch')
            _run(['git', 'worktree', 'add', '--detach', str(worktree), 'HEAD'], cwd=repo_root)
            _run(['git', 'checkout', '--orphan', 'gh-pages'], cwd=worktree)
            # Empty the working tree so the orphan branch starts clean.
            for entry in worktree.iterdir():
                if entry.name == '.git':
                    continue
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()

        try:
            _copy_html_into(worktree, subdir, html_dir)
            if is_latest:
                _copy_html_into(worktree, 'latest', html_dir)
            _write_site_metadata(worktree, parsed_tags, latest_stable)

            _run(['git', 'add', '-A'], cwd=worktree)
            # Skip commit if nothing actually changed.
            status = _run(['git', 'status', '--porcelain'], cwd=worktree, capture=True)
            if not status:
                print('  no changes to commit')
                return

            _run(['git', 'commit', '-m', commit_message], cwd=worktree)

            if dry_run:
                print('  --dry-run: skipping git push')
            else:
                _run(['git', 'push', 'origin', 'gh-pages'], cwd=worktree)
        finally:
            # Always release the worktree so the temp dir can be cleaned.
            # We swallow only expected filesystem/subprocess failures here so
            # cleanup can't mask a real error from the try-block above.
            # `check=False` already prevents non-zero git exits from raising;
            # what's left is git-binary-missing (OSError) and defensive cover
            # for CalledProcessError in case `check=False` is ever removed.
            try:
                _run(
                    ['git', 'worktree', 'remove', '--force', str(worktree)],
                    cwd=repo_root,
                    check=False,
                )
            except (OSError, subprocess.CalledProcessError) as exc:
                print(f'  (worktree cleanup warning: {exc})')


def publish_to_outdir(out_dir, html_dir, subdir, is_latest, parsed_tags, latest_stable):
    """Assemble the full site tree in a plain directory (no git)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    _copy_html_into(out_dir, subdir, html_dir)
    if is_latest:
        _copy_html_into(out_dir, 'latest', html_dir)
    _write_site_metadata(out_dir, parsed_tags, latest_stable)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument(
        '--kind',
        choices=['dev', 'tag'],
        required=True,
        help="'dev' publishes to /dev/; 'tag' publishes to /<tag>/.",
    )
    ap.add_argument('--tag', help='Tag name (e.g. v1.0.1). Required when --kind=tag.')
    ap.add_argument(
        '--html-dir', required=True, help='Path to the built HTML (e.g. aviary/docs/_build/html).'
    )
    ap.add_argument(
        '--repo-root', default=None, help='Repo root (default: cwd). Only used for worktree mode.'
    )
    ap.add_argument(
        '--out-dir',
        default=None,
        help='Write the full site tree to this directory instead '
        'of pushing to gh-pages. Useful for local preview.',
    )
    ap.add_argument(
        '--dry-run',
        action='store_true',
        help='Skip the final `git push`. Ignored when --out-dir is set.',
    )
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if args.kind == 'tag' and not args.tag:
        print('--tag is required when --kind=tag', file=sys.stderr)
        return 2

    html_dir = Path(args.html_dir).resolve()
    if not html_dir.is_dir():
        print(f'HTML dir not found: {html_dir}', file=sys.stderr)
        return 2

    repo_root = Path(args.repo_root).resolve() if args.repo_root else Path.cwd()

    # Determine target subdir + whether this build also refreshes /latest/.
    if args.kind == 'dev':
        subdir = 'dev'
    else:
        subdir = args.tag
    _validate_subdir(subdir)

    # If this run is publishing a specific tag, always consider that tag a
    # candidate — even if `git tag -l` on the runner didn't surface it (e.g.
    # tags weren't fetched into the checkout).
    extra = [args.tag] if args.kind == 'tag' else []
    parsed_tags = get_release_tags(repo_root, extra_tags=extra)

    # After this publish, what will /latest/ point at?
    #   - dev builds: whatever it already was (recomputed from tag list).
    #   - tag builds: the newly-published tag if it's the highest stable release,
    #     otherwise unchanged.
    latest_stable_now = pick_latest_stable(parsed_tags)
    is_latest = args.kind == 'tag' and args.tag == latest_stable_now

    commit_message = f'Publish docs for {subdir}' + (' (also updated latest/)' if is_latest else '')

    if args.out_dir:
        publish_to_outdir(
            out_dir=Path(args.out_dir).resolve(),
            html_dir=html_dir,
            subdir=subdir,
            is_latest=is_latest,
            parsed_tags=parsed_tags,
            latest_stable=latest_stable_now,
        )
        print(f'Local site tree written to {args.out_dir}')
        print(f'Preview with: python -m http.server -d {args.out_dir} 8000')
    else:
        publish_to_worktree(
            repo_root=repo_root,
            html_dir=html_dir,
            subdir=subdir,
            is_latest=is_latest,
            parsed_tags=parsed_tags,
            latest_stable=latest_stable_now,
            dry_run=args.dry_run,
            commit_message=commit_message,
        )
        print('Done.' + (' (dry-run)' if args.dry_run else ''))

    return 0


if __name__ == '__main__':
    sys.exit(main())
