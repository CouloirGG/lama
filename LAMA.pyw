"""Double-click this file to launch LAMA — no console window."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from app import main
main()
