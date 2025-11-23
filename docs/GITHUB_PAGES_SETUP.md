# GitHub Pages Setup Instructions

## Enable GitHub Pages for Doxygen Documentation

Follow these steps to enable automatic Doxygen documentation deployment:

### 1. Enable GitHub Pages in Repository Settings

1. Go to your repository on GitHub: `https://github.com/antshiv/C-Transformer`
2. Click on **Settings** (top right)
3. In the left sidebar, click **Pages** (under "Code and automation")
4. Under **Source**, select:
   - Source: **GitHub Actions** (not "Deploy from a branch")
5. Click **Save**

### 2. Push Changes to GitHub

```bash
git add .
git commit -m "Add Doxygen documentation with GitHub Pages deployment"
git push origin main
```

### 3. Wait for Deployment

1. Go to **Actions** tab in your repository
2. You should see a workflow run called "Deploy Doxygen to GitHub Pages"
3. Wait for it to complete (usually 2-3 minutes)
4. Once completed, your docs will be live at:
   - **https://antshiv.github.io/C-Transformer/**

### 4. Verify Deployment

- Click the documentation badge in your README
- Or visit the URL directly
- You should see your beautiful Doxygen documentation!

## Local Testing

Before pushing, you can test the documentation locally:

```bash
# Generate documentation
./run_doxygen.sh

# Open in browser
firefox docs/html/index.html
# or
xdg-open docs/html/index.html
```

## Automatic Updates

Every time you push to the `main` branch:
1. GitHub Actions automatically runs
2. Doxygen generates fresh documentation
3. Docs are deployed to GitHub Pages
4. Your documentation stays up-to-date!

## Troubleshooting

### Workflow fails?
- Check the Actions tab for error messages
- Ensure Doxyfile is valid (run `./run_doxygen.sh` locally)

### 404 on docs page?
- Verify GitHub Pages is set to "GitHub Actions" (not branch)
- Check that the workflow completed successfully
- Wait a few minutes for DNS propagation

### Documentation not updating?
- Check that your push triggered the workflow (Actions tab)
- Clear browser cache
- Force refresh with Ctrl+Shift+R (or Cmd+Shift+R on Mac)

## Manual Workflow Trigger

You can also manually trigger documentation generation:

1. Go to **Actions** tab
2. Click "Deploy Doxygen to GitHub Pages"
3. Click **Run workflow** button
4. Select branch and click **Run workflow**
