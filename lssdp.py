import importlib
import sys

def check_package(package_name, import_name=None):
    """Check if a package is properly installed"""
    if import_name is None:
        import_name = package_name
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name}")
        return True
    except ImportError as e:
        print(f"❌ {package_name}: {e}")
        return False

required_packages = [
    ("streamlit", "streamlit"),
    ("scikit-learn", "sklearn"),
    ("pandas", "pandas"),
    ("numpy", "numpy"),
    ("spacy", "spacy"),
    ("textblob", "textblob"),
    ("matplotlib", "matplotlib"),
    ("seaborn", "seaborn"),
    ("plotly", "plotly"),
]

print("Checking TextInsight dependencies...")
all_installed = all(check_package(pkg, imp) for pkg, imp in required_packages)

if all_installed:
    print("\n🎉 All dependencies installed successfully!")
    print("Now download the spaCy model: python -m spacy download en_core_web_sm")
else:
    print("\n⚠️ Some dependencies are missing. Please install them using:")
    print("pip install -r requirements.txt")
    sys.exit(1)
