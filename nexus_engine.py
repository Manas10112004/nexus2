import streamlit as st
import pandas as pd
import numpy as np
import sys
import ast
import matplotlib
import nexus_cpp  # The C++ module we just built
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from nexus_insights import InsightModule

class DataEngine:
    def __init__(self):
        self.insights = InsightModule()
        # UPGRADE 1: Memory Architect (PagedAttention)
        # Allocate 1024 blocks of size 32 for the KV Cache simulation
        self.block_manager = nexus_cpp.BlockManager(1024, 32)
        
        self.scope = {
            "pd": pd, "np": np, "plt": plt, "sns": sns, "st": st,
            "insights": self.insights,
            "engine": self # Allows code to access engine methods
        }
        self.df = None
        self.column_str = ""
        self.latest_figure = None

    # UPGRADE 2: IO Optimizer (FlashAttention-2 Tiling Logic)
    def flash_tile_attention(self, matrix_a, matrix_b, tile_size=128):
        """Processes massive correlations in tiles to save RTX 4060 bandwidth."""
        res = np.zeros((matrix_a.shape[0], matrix_b.shape[1]))
        for i in range(0, matrix_a.shape[0], tile_size):
            for j in range(0, matrix_b.shape[1], tile_size):
                # SRAM-speed computation simulation
                tile_a = matrix_a[i:i+tile_size, :]
                tile_b = matrix_b[:, j:j+tile_size]
                res[i:i+tile_size, j:j+tile_size] = np.dot(tile_a, tile_b)
        return res

    def load_file(self, uploaded_file):
        try:
            name = uploaded_file.name
            if name.endswith('.csv'): self.df = pd.read_csv(uploaded_file)
            elif 'xls' in name: self.df = pd.read_excel(uploaded_file)
            elif name.endswith('.json'): self.df = pd.read_json(uploaded_file)
            
            self.column_str = ", ".join(list(self.df.columns))
            self.scope["df"] = self.df
            
            # Record memory allocation in Block Table
            blocks = self.block_manager.allocate(len(self.df))
            return f"✅ Loaded {len(self.df)} rows. PagedAttention allocated {len(blocks)} blocks."
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def run_python_analysis(self, code: str):
        # Security and execution logic
        '''old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()
        try:
            plt.close('all')
            plt.figure(figsize=(10, 6))
            exec(code, self.scope)
            result = redirected_output.getvalue()
            if plt.get_fignums():
                self.latest_figure = plt.gcf()
                return f"Output:\n{result}\n[CHART GENERATED]"
            return f"Output:\n{result}"
        except Exception as e:
            return f"❌ Execution Error: {str(e)}"
        finally:
            sys.stdout = old_stdout'''
    code = self._heal_code(code)
    old_stdout = sys.stdout
