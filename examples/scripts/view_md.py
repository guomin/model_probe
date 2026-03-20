#!/usr/bin/env python3
"""Markdown终端渲染器"""
import sys
sys.path.insert(0, "/mnt/hdsd1/guomin/projects/model_probe")

from rich.console import Console
from rich.markdown import Markdown
import os

def render_md(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    
    console = Console()
    md = Markdown(content)
    console.print(md)

if __name__ == "__main__":
    filepath = sys.argv[1] if len(sys.argv) > 1 else "analysis_report.md"
    render_md(filepath)
