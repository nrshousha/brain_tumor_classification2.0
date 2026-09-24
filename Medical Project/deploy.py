import os
import setuptools
from app import app

if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 NeuroAI Server Running: http://127.0.0.1:3000")
    print("="*60 + "\n")
    app.run(host='127.0.0.1', port=3000, debug=False)