# Interactive network — deployment notes

Static page, no build step, no external CDN. Three files:

    index.html                     the visualisation
    data.js                        window.VIZ_DATA — nodes, book-to-book edges, 3,046 pairs
    matrix_classifications.csv     the downloadable table the Download button serves

## Deploy to datasciencemagic.com

The site is GitHub Pages (repo `sspackman99.github.io`, Chirpy theme, built by
`.github/workflows/pages-deploy.yml`). Static folders are copied verbatim, and this
page contains no Liquid tags, so it needs no Jekyll configuration.

    cp -r viz ~/work/sspackman99.github.io/quote-network
    cd ~/work/sspackman99.github.io
    git add quote-network
    git commit -m "Add scripture quotation network"
    git push

Then: https://datasciencemagic.com/quote-network/

To link it from the blog, either add a Chirpy post whose body is a link, or an
iframe:

    <iframe src="/quote-network/" style="width:100%;height:820px;border:0"></iframe>

## Regenerate after a new classification run

    .venv/bin/python build_exports.py        # writes matrix_classifications.csv + viz/data.js
    cp viz/index.html viz/data.js viz/matrix_classifications.csv \
       ~/work/sspackman99.github.io/quote-network/

## What the page does

- Nodes are books (60 of them), sized by how many classified connections touch them,
  coloured by corpus.
- Lines are book-to-book edges, width by count, colour by the dominant label.
- A ring around a book counts repetition inside that book (Numbers → Numbers, D&C → D&C).
- Click a book: its neighbours, ranked. Click a neighbour: the actual passage pairs,
  with the classifier's explanation. Click a pair: the evidence it quoted.
- Filters: corpus, label, cosine floor, minimum connection count, free-text search.
