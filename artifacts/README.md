# Analytics upgrade mockups

## View immediately

After pulling this branch, open [`analytics-upgrades-mockup.svg`](./analytics-upgrades-mockup.svg) directly in any browser. It is a standalone, scalable image artifact and does not require the application or any dependencies.

## View the interactive responsive version

The implementation also includes `static/analytics_methodology_mockup.html`. When the app is running, visit either the interactive mockup or the directly served image artifact:

```text
http://localhost:5000/static/analytics_methodology_mockup.html
http://localhost:5000/static/analytics-upgrades-mockup.svg
```

Or open the file directly from the repository. All mockup-specific styling is embedded, while `/static/dashboard.css` supplies the site font and shared defaults when served by Flask.

## Download/share

Use the browser's **Save image as…** action on the SVG. Because it is vector-based, it remains sharp when exported to PNG or PDF at any size.

## If your local path says the file does not exist

The artifact was added in commit `3d1bee9`. Your local checkout must contain that commit. Fetch/check out the PR branch (or cherry-pick that commit) before opening the path. Verify with:

```bash
git log --oneline --all -- artifacts/analytics-upgrades-mockup.svg
test -f artifacts/analytics-upgrades-mockup.svg && echo "artifact ready"
```
