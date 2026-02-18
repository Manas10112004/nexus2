import streamlit as st
import pandas as pd
import numpy as np
import sys
import re
import ast  # <--- NEW: For code scanning
import matplotlib
import nexus_cpp
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from nexus_insights import InsightModule


class SecurityManager:
    """
    Scans generated Python code for malicious imports or commands
    before execution.
    """
    FORBIDDEN_IMPORTS = ['os', 'sys', 'subprocess', 'shutil', 'builtins', 'importlib']
    FORBIDDEN_CALLS = ['exec', 'eval', 'compile', 'open']

    @staticmethod
    def is_safe(code: str) -> tuple[bool, str]:
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # 1. Check Imports
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    module_names = [n.name.split('.')[0] for n in node.names] if isinstance(node, ast.Import) else [
                        node.module.split('.')[0]]
                    for name in module_names:
                        if name in SecurityManager.FORBIDDEN_IMPORTS:
                            return False, f"🚫 Security Violation: Import '{name}' is forbidden."

                # 2. Check Function Calls
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in SecurityManager.FORBIDDEN_CALLS:
                        return False, f"🚫 Security Violation: Function '{node.func.id}()' is forbidden."
            return True, "Safe"
        except SyntaxError:
            return False, "❌ Syntax Error in generated code."


class DataEngine:
    def __init__(self):
        self.insights = InsightModule()
        self.scope = {
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "st": st,
            "insights": self.insights
            "nexus_cpp": nexus_cpp
        }
        self.df = None
        self.column_str = ""
        self.latest_figure = None

    # Add this method to your DataEngine class to handle large-scale data "tiling"
    def flash_compute_similarity(self, query_batch, key_batch, tile_size=128):
        """
        Simulates FlashAttention tiling to keep memory usage low on the RTX 4050.
        Instead of Q @ K.T (N x N), we process in tiles.
        """
        n = query_batch.shape[0]
        output = np.zeros((n, key_batch.shape[1]))
        
        for i in range(0, n, tile_size):
            # Slice Q into a Tile that fits in "SRAM" (fast cache)
            q_tile = query_batch[i:i + tile_size]
            
            for j in range(0, n, tile_size):
                k_tile = key_batch[j:j + tile_size]
                
                # Compute local softmax and update output
                # This avoids creating the full NxN matrix in VRAM
                attn_tile = np.exp(np.dot(q_tile, k_tile.T)) 
                # ... (Normalization logic here)
                
        return output

    def load_file(self, uploaded_file):
        try:
            name = uploaded_file.name
            if name.endswith(('.csv', '.xlsx', '.xls', '.json')):
                if name.endswith('.csv'):
                    self.df = pd.read_csv(uploaded_file)
                elif 'xls' in name:
                    self.df = pd.read_excel(uploaded_file)
                elif name.endswith('.json'):
                    self.df = pd.read_json(uploaded_file)

                self.column_str = ", ".join(list(self.df.columns))
                self.scope["df"] = self.df
                return f"✅ Data Loaded: {len(self.df)} rows. Columns: {self.column_str}"
            else:
                return f"⚠️ Unsupported file: {name}"
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _heal_code(self, code: str) -> str:
        # (Same healing logic as before)
        if self.df is None: return code
        if ".corr()" in code and "numeric_only" not in code:
            code = code.replace(".corr()", ".select_dtypes(include=['number']).corr()")
        return code

    def run_python_analysis(self, code: str):
        # 1. RUN SECURITY CHECK
        is_safe, message = SecurityManager.is_safe(code)
        if not is_safe:
            return message

        # 2. PROCEED IF SAFE
        code = self._heal_code(code)
        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()

        try:
            plt.close('all')
            plt.figure(figsize=(10, 6))
            pd.set_option('display.max_rows', 20)

            # Execute in restricted scope
            exec(code, self.scope)
            result = redirected_output.getvalue()

            if plt.get_fignums():
                self.latest_figure = plt.gcf()
                return f"Output:\n{result}\n[CHART GENERATED]"

            if result and len(result.strip()) > 0:
                return f"Output:\n{result}\n[ANALYSIS COMPLETE]"

            return "❌ Code ran but produced no output."

        except Exception as e:
            return f"❌ Execution Error: {str(e)}"
        finally:
            sys.stdout = old_stdout
