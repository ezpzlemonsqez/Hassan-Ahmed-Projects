# Portfolio redesign

The audience is recruiters and people assessing Hassan's skills. The design should make the person, qualifications, project scope and evidence easy to understand. It should feel like a considered mathematical portfolio.

## What the supplied website did

The homepage combined its content, CSS and a small JavaScript card-navigation handler in one file. Its colour variables defined a dark navy palette with blue and purple accents. Radial gradients, translucent panels, large shadows and repeated rounded borders created most of its visual identity. Nested grids supplied the About blocks and two project columns; media queries stacked them on smaller screens.

Each project had a PDF reader and a code overview. The PDF iframe loaded a local report, with an explicit link for browsers that could not embed it. Code overviews embedded individual HTML source previews. Those previews fetched the original Python, R or C++ files and passed the resulting text to Highlight.js. Some tried historical alternative filenames. The MSc project instead embedded an existing exported notebook. These are useful, intentional distinctions.

The old card handler made a whole row clickable while excluding its inner links. Its keyboard handler supplied Enter and Space behaviour because the row itself was not a native link. The new design uses a native title link with a CSS hit area, so browser link actions and Enter navigation work without a homepage script. Report and Code links remain independent.

## Design decisions

- Use warm paper, dark ink and a restrained green accent. Fine rules, spacing and typography establish hierarchy throughout the homepage and project readers.
- Keep Hassan Ahmed as the main heading. Put the MSc, First-Class BSc and location near the introduction so readers can place the work in context.
- Retain About before Projects, the three centred About blocks, quantitative research on the left, applied finance/risk/analytics on the right, and Contact at the end.
- Retain all eight projects, their ordering, original descriptions and existing routes. Put language and subject context above each title. Keep reports and code one click away.
- Use one precisely defined mathematical illustration: the bivariate standard normal density on [-3, 3] in each coordinate. Its caption identifies it as an illustration, not a project finding. The projected mesh is sampled from exp(-(x²+y²)/2)/(2π).
- Avoid decorative statistics, return claims, animated tickers and labels implying professional experience that the projects do not establish.

## Implementation and preservation

The redesign changes the presentation pages and shared web assets. Original research source, data, PDFs, notebook and exported notebook are retained. Shared CSS replaces repeated per-page styles, and one source loader replaces repeated fetch/highlight code.

Source content is inserted using `textContent`. Fetch errors and highlighting errors are handled separately: unavailable highlighting leaves the source readable. Copy is progressively enabled only when the browser supports it, and Raw links provide direct access. Fonts and Highlight.js are local; the untouched notebook export has its original dependencies.

Navigation uses semantic landmarks, headings, real links, visible focus outlines and a skip link. Layouts adapt to narrow screens. There are no entrance animations; reduced-motion preferences disable smooth scrolling and transitions. PDF embedding still depends on the visitor's browser, with a prominent direct PDF link available.

See the README for local preview and maintenance instructions.
