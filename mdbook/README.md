# Developing this book

Make sure you have [mdBook](https://github.com/rust-lang/mdBook) and the
preprocessors the book uses
([linkcheck](https://github.com/Michael-F-Bryan/mdbook-linkcheck),
[mermaid](https://github.com/badboy/mdbook-mermaid) and
[admonish](https://github.com/tommilligan/mdbook-admonish)). These are the
versions that the `Deploy mdBook` GitHub workflow uses:

    cargo install mdbook --version 0.4.43
    cargo install mdbook-linkcheck --version 0.7.7
    cargo install mdbook-mermaid --version 0.13.0
    cargo install mdbook-admonish --version 1.16.0

`cd` to the `mdbook` directory and run:

    mdbook-admonish install
    mdbook serve

The book is published to
[GitHub pages](https://mwatelescope.github.io/mwa_hyperdrive) from the `main`
branch, so it describes `main`, not the latest release.

# Mermaid

mermaid-init.js and mermaid.min.js were provided by mdbook-mermaid, v0.11.0.
mdbook-mermaid is licensed under the MPL-2.0, the same license used by
mwa_hyperdrive.
