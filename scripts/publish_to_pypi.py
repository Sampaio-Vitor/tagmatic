#!/usr/bin/env python3
"""
Script to help publish Tagmatic to PyPI.

This script automates the process of building and uploading the package to PyPI.
Make sure you have the required tools installed:
    pip install build twine

Usage:
    python scripts/publish_to_pypi.py --test    # Upload to TestPyPI
    python scripts/publish_to_pypi.py --prod    # Upload to PyPI
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(command, description):
    """Run a shell command and handle errors."""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"Error: {e.stderr}")
        return False


def clean_build_artifacts():
    """Clean previous build artifacts."""
    print("\n🧹 Cleaning previous build artifacts...")
    build_dirs = ["build", "dist", "*.egg-info"]
    for pattern in build_dirs:
        subprocess.run(f"rm -rf {pattern}", shell=True)
    print("✅ Build artifacts cleaned")


def build_package():
    """Build the package."""
    return run_command("python -m build", "Building package")


def upload_to_testpypi():
    """Upload package to TestPyPI."""
    return run_command(
        "python -m twine upload --repository testpypi dist/*",
        "Uploading to TestPyPI"
    )


def upload_to_pypi():
    """Upload package to PyPI."""
    return run_command(
        "python -m twine upload dist/*",
        "Uploading to PyPI"
    )


def check_dependencies():
    """Check if required dependencies are installed."""
    print("🔍 Checking required dependencies...")
    
    required_packages = ["build", "twine"]
    missing_packages = []
    
    for package in required_packages:
        try:
            subprocess.run(f"python -m {package} --help", 
                         shell=True, check=True, capture_output=True)
        except subprocess.CalledProcessError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Install them with: pip install build twine")
        return False
    
    print("✅ All required dependencies are installed")
    return True


def main():
    parser = argparse.ArgumentParser(description="Publish Tagmatic to PyPI")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--test", action="store_true", help="Upload to TestPyPI")
    group.add_argument("--prod", action="store_true", help="Upload to PyPI")
    
    args = parser.parse_args()
    
    # Change to project root directory
    project_root = Path(__file__).parent.parent
    print(f"📁 Working in directory: {project_root}")
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Clean previous builds
    clean_build_artifacts()
    
    # Build package
    if not build_package():
        sys.exit(1)
    
    # Upload package
    if args.test:
        print("\n🚀 Uploading to TestPyPI...")
        if not upload_to_testpypi():
            sys.exit(1)
        print("\n✅ Package successfully uploaded to TestPyPI!")
        print("You can install it with:")
        print("pip install --index-url https://test.pypi.org/simple/ tagmatic")
    
    elif args.prod:
        print("\n🚀 Uploading to PyPI...")
        confirmation = input("Are you sure you want to upload to PyPI? (yes/no): ")
        if confirmation.lower() != "yes":
            print("Upload cancelled.")
            sys.exit(0)
        
        if not upload_to_pypi():
            sys.exit(1)
        print("\n🎉 Package successfully uploaded to PyPI!")
        print("You can install it with:")
        print("pip install tagmatic")


if __name__ == "__main__":
    main()
