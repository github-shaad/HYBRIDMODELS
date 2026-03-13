"""
Simple Class to save data/models/statistics/plots.
"""
from config.config import *
import numpy as np
import openpyxl
from plots import *
class StorageManager:
    def __init__(self):
        pass

    def store_predictions(self, data, prefix:str, modelspec:str):
        """
        Store Predictions \n
        prefix : Model or Portfolio \n
        modelspec : Which model's predictions being stored
        """
        predictions_path = PREDICTIONS_DIR / f"{prefix}_predictions" / modelspec
        np.save(predictions_path, data)
    
    def store_statistics(self, prefix, stat, model, value):
        path = None
        if prefix == "data":
            path = DATA_STATISTICS
        elif prefix == "model":
            path = MODEL_STATISTICS
        elif prefix == "portfolio":
            path = PORTFOLIO_STATISTICS
        else:
            raise ValueError("Wrong prefix type. Not data/model/portfolio")

        file_path = path / f"{prefix}_statistics.xlsx"
        
        workbook = openpyxl.load_workbook(file_path)
        sheet = workbook.active

        header_map = {cell.value: cell.column for cell in sheet[1] if cell.value is not None}



        if "Model" not in header_map or stat not in header_map:
            raise ValueError(f"Non-existent metrics or excel sheet structure.\nCheck excel file located at {file_path}")

        model_col = header_map["Model"]    
        stat_col = header_map[stat]

        target_row = None
        for row in range(2, sheet.max_row+1):
            if sheet.cell(row=row, column=model_col).value == model:
                target_row = row

        if target_row is None:
            target_row = sheet.max_row + 1
            sheet.cell(row=target_row, column=model_col, value=model)

        sheet.cell(row=target_row, column=stat_col, value=value)

        workbook.save(file_path)
    
    def store_figures(self, plot, prefix, modelspec):
        path = FIGURES_DIR
        file_path = path / f"{prefix}_figures" / f"{modelspec}.png"
        plot.fig.savefig(file_path)
