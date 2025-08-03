"""
Deployment preparation script for Streamlit Cloud.
This script ensures the app is ready for deployment.
"""

import subprocess
import sys
import os
import logging
from health_check import main as run_health_check

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def check_git_status():
    """Check if we're in a git repository and files are committed."""
    try:
        # Check if we're in a git repo
        result = subprocess.run(['git', 'status', '--porcelain'], 
                              capture_output=True, text=True, check=True)
        
        if result.stdout.strip():
            logger.warning("⚠️ You have uncommitted changes:")
            logger.warning(result.stdout)
            logger.warning("Consider committing changes before deployment")
        else:
            logger.info("✅ Git repository is clean")
        
        return True
    except subprocess.CalledProcessError:
        logger.warning("⚠️ Not in a git repository or git not available")
        return False
    except FileNotFoundError:
        logger.warning("⚠️ Git not found in PATH")
        return False

def create_deployment_checklist():
    """Create a deployment checklist file."""
    checklist = """# Deployment Checklist for Streamlit Cloud

## Pre-deployment Steps
- [ ] All dependencies are listed in requirements.txt with versions
- [ ] Model file (svm_digit_model.pkl) is present and valid
- [ ] Health check passes (run: python health_check.py)
- [ ] App runs locally without errors (run: streamlit run app.py)
- [ ] All files are committed to git repository
- [ ] Repository is pushed to GitHub

## Streamlit Cloud Setup
- [ ] Connect GitHub account to Streamlit Cloud
- [ ] Select the repository containing this app
- [ ] Set main file path to: app.py
- [ ] Deploy the app

## Post-deployment Verification
- [ ] App loads successfully on Streamlit Cloud
- [ ] Image upload functionality works
- [ ] Model predictions are accurate
- [ ] Error handling works as expected
- [ ] UI is responsive and looks correct

## Troubleshooting
If deployment fails:
1. Check the logs in Streamlit Cloud dashboard
2. Verify all dependencies are correctly specified
3. Ensure model file is included in the repository
4. Run health_check.py locally to identify issues

## Performance Optimization
- Model is cached using @st.cache_resource
- Images are processed efficiently
- Error handling prevents crashes
- UI is optimized for mobile and desktop
"""
    
    with open("DEPLOYMENT_CHECKLIST.md", "w") as f:
        f.write(checklist)
    
    logger.info("✅ Created DEPLOYMENT_CHECKLIST.md")

def optimize_for_deployment():
    """Perform deployment optimizations."""
    
    # Check if large data files should be excluded
    large_files = []
    for file in ['mnist_train.csv', 'mnist_test.csv']:
        if os.path.exists(file):
            size_mb = os.path.getsize(file) / (1024 * 1024)
            if size_mb > 100:  # Files larger than 100MB
                large_files.append((file, size_mb))
    
    if large_files:
        logger.warning("⚠️ Large data files detected:")
        for file, size in large_files:
            logger.warning(f"  {file}: {size:.1f} MB")
        logger.warning("Consider using Git LFS or excluding from repository")
        logger.warning("The model file (svm_digit_model.pkl) is sufficient for deployment")
    
    # Check model file size
    if os.path.exists('svm_digit_model.pkl'):
        model_size = os.path.getsize('svm_digit_model.pkl') / (1024 * 1024)
        logger.info(f"Model file size: {model_size:.1f} MB")
        
        if model_size > 100:
            logger.warning("⚠️ Model file is large, consider model compression")

def main():
    """Run deployment preparation."""
    logger.info("🚀 Preparing for Streamlit Cloud deployment...")
    
    # Run health checks first
    logger.info("\n1. Running health checks...")
    if not run_health_check():
        logger.error("❌ Health checks failed. Fix issues before deploying.")
        return False
    
    # Check git status
    logger.info("\n2. Checking git status...")
    check_git_status()
    
    # Create deployment checklist
    logger.info("\n3. Creating deployment checklist...")
    create_deployment_checklist()
    
    # Optimization checks
    logger.info("\n4. Running deployment optimizations...")
    optimize_for_deployment()
    
    logger.info("\n🎉 Deployment preparation complete!")
    logger.info("\nNext steps:")
    logger.info("1. Review DEPLOYMENT_CHECKLIST.md")
    logger.info("2. Commit and push changes to GitHub")
    logger.info("3. Deploy on Streamlit Cloud")
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)