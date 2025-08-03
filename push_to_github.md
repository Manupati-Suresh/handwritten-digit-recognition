# Push to GitHub Instructions

After creating your repository on GitHub, run these commands:

```bash
# Add the remote repository (replace 'yourusername' with your GitHub username)
git remote add origin https://github.com/yourusername/handwritten-digit-recognition.git

# Push the code to GitHub
git branch -M main
git push -u origin main
```

## Alternative: If you want to use a different repository name
```bash
# For a different name, replace 'handwritten-digit-recognition' with your chosen name
git remote add origin https://github.com/yourusername/your-repo-name.git
git branch -M main
git push -u origin main
```

## After pushing to GitHub:

1. Your code will be available on GitHub
2. You can then deploy it on Streamlit Cloud by:
   - Going to [share.streamlit.io](https://share.streamlit.io)
   - Connecting your GitHub account
   - Selecting your repository
   - Setting the main file to: `app.py`
   - Clicking "Deploy"

## Verification:
- Run `python health_check.py` to ensure everything is working
- Your app should be ready for deployment!