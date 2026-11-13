# Security

## What it does

- **Nothing over the network.** The command line and the page compute everything locally from built-in data and your
  inputs; no request is made, nothing is sent.
- **A config.json is read as data**: parsed as JSON, only its numbers used; nothing in it is run.
- **The page** loads only its own files (a Content-Security-Policy: `default-src 'none'`, scripts and styles from the
  page itself, no inline script), sets every text with `textContent`, and keeps your settings in the link (the part
  after `#`, which browsers do not send to the server).
- **No runtime dependencies.** The only development dependencies are the TypeScript checker and Node's types.

## Reporting a vulnerability

Please open a private security advisory on GitHub (Security → Report a vulnerability) rather than a public issue.
