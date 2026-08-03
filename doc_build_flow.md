::: mermaid
flowchart TD
    %% ── Triggers ─────────────────────────────────────────────
    subgraph TRIG["Triggers"]
        A1["push to main<br/>(or IA966-versioned-doc, temp)"]:::trig
        A2["push tag v*.*.*"]:::trig
        A3["workflow_dispatch<br/>(tag input)"]:::trig
        A4["pull_request → main"]:::trig
    end

    %% ── Workflows ────────────────────────────────────────────
    A1 --> W1["test_docs.yml"]:::wf
    A4 --> W1
    A2 --> W2["publish_docs.yml"]:::wf
    A3 --> W2

    %% ── Shared build stages ──────────────────────────────────
    W1 --> B1["Checkout repo<br/>+ prepare_environment (pixi)"]:::step
    W2 --> B2["Checkout at tag<br/>fetch-depth: 0<br/>+ prepare_environment"]:::step

    B1 --> LINT["Lint notebooks<br/>check_jupyter_output_linting.py"]:::step
    LINT --> BUILD1["build_book.sh<br/>→ aviary/docs/_build/html"]:::step
    B2 --> BUILD2["build_book.sh<br/>→ aviary/docs/_build/html"]:::step

    %% ── PR path stops before publish ─────────────────────────
    BUILD1 --> PR{"push event<br/>and repo == hschilling/Aviary<br/>and ref in (main, dev-branch)?"}:::decision
    PR -- "no (PR / wrong repo)" --> STOP1["Stop — build only,<br/>no publish"]:::stop
    PR -- "yes" --> PUB1["publish_docs.py --kind dev"]:::script

    BUILD2 --> PUB2["publish_docs.py --kind tag<br/>--tag vX.Y.Z"]:::script

    %% ── publish_docs.py internals ────────────────────────────
    subgraph PY["publish_docs.py — same code, both paths"]
        direction TB
        P1["Enumerate git tags<br/>packaging.version.Version sort"]:::pystep
        P2["Pick highest stable<br/>→ latest_stable"]:::pystep
        P3["git worktree add gh-pages<br/>(or orphan if branch missing)"]:::pystep
        P4["rm & copy _build/html<br/>→ worktree/&lt;subdir&gt;/"]:::pystep
        P5{"is this tag<br/>== latest_stable?"}:::decision
        P6["also refresh<br/>worktree/latest/"]:::pystep
        P7["Regenerate versions.json<br/>from dirs actually present"]:::pystep
        P8["Regenerate index.html<br/>landing page"]:::pystep
        P9["Ensure .nojekyll exists"]:::pystep
        P10["git commit + push<br/>origin gh-pages"]:::pystep

        P1 --> P2 --> P3 --> P4 --> P5
        P5 -- yes --> P6 --> P7
        P5 -- no --> P7
        P7 --> P8 --> P9 --> P10
    end

    PUB1 --> PY
    PUB2 --> PY

    %% ── Resulting site ──────────────────────────────────────
    P10 --> SITE["gh-pages branch<br/>on hschilling/Aviary"]:::branch

    subgraph LAYOUT["Published site layout"]
        direction TB
        L1["/ → index.html<br/>(landing + version picker)"]:::site
        L2["/versions.json"]:::site
        L3["/dev/"]:::site
        L4["/latest/"]:::site
        L5["/v1.0.1/"]:::site
        L6["/v1.0.0/ …"]:::site
    end

    SITE --> LAYOUT
    LAYOUT --> URL["https://hschilling.github.io/Aviary/"]:::url

    %% ── Styling ─────────────────────────────────────────────
    classDef trig     fill:#e8f0ff,stroke:#4b6bd6,color:#1a1a1a;
    classDef wf       fill:#ffe8d6,stroke:#c46a1c,color:#1a1a1a,font-weight:bold;
    classDef step     fill:#f5f5f5,stroke:#888,color:#1a1a1a;
    classDef script   fill:#e6f4ea,stroke:#2f8f3f,color:#1a1a1a,font-weight:bold;
    classDef pystep   fill:#eef7ea,stroke:#4a9b5a,color:#1a1a1a;
    classDef decision fill:#fff4c2,stroke:#a68300,color:#1a1a1a;
    classDef stop     fill:#f0e0e0,stroke:#a04040,color:#1a1a1a;
    classDef branch   fill:#e0e0f5,stroke:#5a5a99,color:#1a1a1a;
    classDef site     fill:#fafafa,stroke:#666,color:#1a1a1a;
    classDef url      fill:#d6ebff,stroke:#2f6fb8,color:#1a1a1a,font-weight:bold;
:::
