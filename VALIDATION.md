# Website verification

Verified against the supplied `Hassan-Ahmed-Projects.zip`.

| Check | Result |
| --- | --- |
| Homepage widths | 320, 390, 768, 1024 and 1440 pixels; no horizontal page overflow. |
| Project readers | All 8 report pages and 8 code pages load, have the correct project title and current resource indicator, and fit desktop and narrow mobile widths. |
| Report files | All 8 PDF URLs respond successfully with the original PDF content. |
| Source previews | All 23 fetch their intended original file and display matching source text with syntax highlighting. Browser line-ending normalisation is allowed. |
| Row navigation | Clicking the project row opens its report; its separate Code link opens its code page. Native Enter navigation works. |
| Clipboard | Copy returns the complete source text. |
| Highlighting unavailable | Blocking the highlighter leaves plain source readable. |
| Source unavailable | A clear message and direct source link replace the loading state. |
| JavaScript disabled | Homepage project navigation still works; source previews offer direct links. |
| Reduced motion | Smooth scrolling is disabled when reduced motion is requested. |
| Project preservation | No original files removed. All research source, data, PDFs, notebook and notebook export retain their original bytes. |

Desktop, mobile and code-reader screenshots were inspected. The browser runtime available for automated checks does not display the native PDF plug-in, so PDF verification covers file integrity, successful loading and direct-file links rather than the plug-in's visual rendering. The original native-PDF embedding approach is retained, with a prominent Open PDF action for browsers that need it.

No live GitHub deployment is part of this package.
