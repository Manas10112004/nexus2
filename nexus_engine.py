import streamlit as st
import pandas as pd
import numpy as np
import sys
import re
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO


class DataEngine:
    """
    The Execution Engine.
    Handles data loading, code sanitization, and figure capture.
    """

    def __init__(self):
        self.scope = {
            "pd": pd,
            "np": np,
            "plt": plt,
            "sns": sns,
            "st": st
        }
        self.df = None
        self.column_str = ""
        self.latest_figure = None  # Buffer to store the plot

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

                # Save columns for the Cheat Sheet
                self.column_str = ", ".join(list(self.df.columns))
                self.scope["df"] = self.df
                return f"✅ Data Loaded: {len(self.df)} rows. Columns: {self.column_str}"

            elif name.endswith(('.txt', '.py', '.md', '.log', '.yaml')):
                stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
                self.file_content = stringio.read()
                self.scope["file_content"] = self.file_content
                return f"✅ Text Loaded: {len(self.file_content)} chars."

            else:
                return f"⚠️ Binary file '{name}' (Limited Access)."
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def _heal_code(self, code: str) -> str:
        """Autocorrects common DataFrame errors before execution."""
        if self.df is None: return code

        # 1. FIX: Force numeric_only for correlations/means to prevent crashes on strings
        if ".corr()" in code and "numeric_only" not in code:
            code = code.replace(".corr()", ".select_dtypes(include=['number']).corr()")
        if ".mean()" in code and "numeric_only" not in code:
            code = code.replace(".mean()", ".mean(numeric_only=True)")

        # 2. FIX: Auto-correct column name typos (case-insensitive)
        real_cols = list(self.df.columns)
        col_map = {c.lower(): c for c in real_cols}
        pattern = r"df\[['\"](.*?)['\"]\]"

        def replace_match(match):
            col_name = match.group(1)
            lower_name = col_name.lower()
            if col_name not in real_cols and lower_name in col_map:
                correct_name = col_map[lower_name]
                return f"df['{correct_name}']"
            return match.group(0)

        healed_code = re.sub(pattern, replace_match, code)
        return healed_code

    def run_python_analysis(self, code: str):
        """Executes code and captures figures/output."""
        code = self._heal_code(code)

        old_stdout = sys.stdout
        redirected_output = sys.stdout = StringIO()

        try:
            plt.clf()  # Clear old plots
            exec(code, self.scope)
            result = redirected_output.getvalue()

            # 1. Check for Charts (Save to buffer, don't show yet)
            if plt.get_fignums():
                self.latest_figure = plt.gcf()
                return f"Output:\n{result}\n[CHART GENERATED]"

            # 2. Check for Text Output
            if result and len(result.strip()) > 0:
                return f"Output:\n{result}\n[ANALYSIS COMPLETE]"

            return f"Output:\n{result}" if result else "Code executed successfully."

        except KeyError as e:
            cols = self.column_str if self.column_str else "Data Not Loaded"
            return f"❌ Column Error: {str(e)}\n💡 AVAILABLE COLUMNS: {cols}"

        except Exception as e:
            return f"❌ Execution Error: {str(e)}"
        finally:
            sys.stdout = old_stdout