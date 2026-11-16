# data/

Your own documents go here: PDF, DOCX, PPTX, Markdown, TXT or HTML.
Scanned PDFs work too; pages without a text layer are OCR'd locally.

Then either click **Rescan data folder** in the web UI's Library tab, or run:

```
cd backend
..\.venv\Scripts\python.exe -m athena.index_folder
```

Files uploaded or fetched from a URL in the UI are saved in `data/uploads/`.

Everything here except this README is git-ignored. Documents never get committed,
and nothing leaves your computer. Still, only index files you're comfortable having
in a local search index; avoid ID cards, offer letters and other people's personal data.
