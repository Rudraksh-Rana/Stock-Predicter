#!/usr/bin/env python
"""
Quick setup script for LSTM Stock Predictor.
Run with: python setup.py

This script:
1. Creates necessary directories
2. Installs required dependencies
3. Downloads initial data
4. Validates the installation
"""

import sys
import subprocess
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def run_command(cmd, description):
    """Run a shell command and return success status."""
    logger.info(f"\n{description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=False)
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed: {description}")
        return False

def check_python_version():
    """Check Python version is 3.11+."""
    if sys.version_info < (3, 11):
        logger.error(f"Python 3.11+ required. You have {sys.version_info.major}.{sys.version_info.minor}")
        sys.exit(1)
    logger.info(f"✓ Python {sys.version_info.major}.{sys.version_info.minor} detected")

def create_directories():
    """Create necessary directories."""
    dirs = [
        'data/raw',
        'models/plots',
        'mlruns'
    ]
    for d in dirs:
        Path(d).mkdir(parents=True, exist_ok=True)
    logger.info(f"✓ Created directories: {', '.join(dirs)}")

def install_requirements():
    """Install Python requirements."""
    if not run_command('pip install -q -r requirements.txt', 'Installing dependencies'):
        logger.warning("Dependency installation had issues, but continuing...")

def validate_imports():
    """Validate all key imports work."""
    logger.info("\nValidating imports...")
    imports = [
        ('torch', 'PyTorch'),
        ('pandas', 'Pandas'),
        ('yfinance', 'yfinance'),
        ('fastapi', 'FastAPI'),
        ('streamlit', 'Streamlit'),
    ]
    
    all_ok = True
    for module, name in imports:
        try:
            __import__(module)
            logger.info(f"  ✓ {name}")
        except ImportError:
            logger.error(f"  ✗ {name} - NOT INSTALLED")
            all_ok = False
    
    return all_ok

def print_next_steps():
    """Print next steps for user."""
    logger.info("\n" + "="*60)
    logger.info("SETUP COMPLETE!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("1. Train the model:")
    logger.info("   python src/train_main.py")
    logger.info("\n2. (In another terminal) Start the API:")
    logger.info("   uvicorn api.main:app --reload")
    logger.info("\n3. (In another terminal) Launch the dashboard:")
    logger.info("   streamlit run app/streamlit_app.py")
    logger.info("\n4. (Optional) View MLflow UI:")
    logger.info("   mlflow ui")
    logger.info("="*60)

def main():
    """Run full setup."""
    logger.info("LSTM Stock Predictor - Setup Script")
    logger.info("="*60)
    
    check_python_version()
    create_directories()
    install_requirements()
    
    if not validate_imports():
        logger.error("Some dependencies failed to install. Please check requirements.txt")
        sys.exit(1)
    
    print_next_steps()

if __name__ == "__main__":
    main()
