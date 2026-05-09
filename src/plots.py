import matplotlib.pyplot as plt
import numpy as np
import math
from matplotlib.ticker import PercentFormatter

class MultiPlot:
    def __init__(self, title, true_color='blue', pred_color='red'):
        self.title = title
        self.true_color = true_color
        self.pred_color = pred_color
        self.fig = None

    def plot(self, data_dict, true_label, pred_label, cols=5, figsize_per_plot=(5, 4)):
        """
        data_dict: dict { "TICKER": (true_values, predicted_values) }
        cols: Number of columns in the grid
        """
        n_plots = len(data_dict)
        rows = math.ceil(n_plots / cols)

        fig, axes = plt.subplots(
            rows, cols, 
            figsize=(cols * figsize_per_plot[0], rows * figsize_per_plot[1]),
            constrained_layout=True
        )
        fig.suptitle(self.title, fontsize=16, fontweight='bold')

        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        for i, (ticker, (true_val, pred_val)) in enumerate(data_dict.items()):
            ax = axes_flat[i]
            
            # Plotting lines
            ax.plot(true_val, color=self.true_color, label=true_label, linewidth=1.5)
            ax.plot(pred_val, color=self.pred_color, label=pred_label, linestyle='--', linewidth=1.7)
            ax.set_title(f"Ticker: {ticker}", fontsize=12)
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.grid(alpha=0.3)
            ax.legend(loc='best')

        # Hide unused subplots if n_plots is not a perfect multiple of cols
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis('off')
        
        self.fig = fig
    
    def show(self):
        if self.fig is not None:
            plt.show()
        else:
            raise ValueError("Figure not created!!")
        
class TripleMultiPlot:
    def __init__(self, title, true_color='blue', pred_color='red', ml_pred_color='orange'):
        self.title = title
        self.true_color = true_color
        self.pred_color = pred_color
        self.ml_pred_color = ml_pred_color
        self.fig = None

    def plot(self, data_dict, true_label, pred_label, ml_pred_label, cols=5, figsize_per_plot=(5, 4)):
        """
        data_dict: dict { "TICKER": (true_values, predicted_values) }
        cols: Number of columns in the grid
        """
        n_plots = len(data_dict)
        rows = math.ceil(n_plots / cols)

        fig, axes = plt.subplots(
            rows, cols, 
            figsize=(cols * figsize_per_plot[0], rows * figsize_per_plot[1]),
            constrained_layout=True
        )
        fig.suptitle(self.title, fontsize=16, fontweight='bold')

        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        for i, (ticker, (true_val, pred_val, ml_pred_val)) in enumerate(data_dict.items()):
            ax = axes_flat[i]
            
            # Plotting lines
            ax.plot(true_val, color=self.true_color, label=true_label, linewidth=1.5)
            ax.plot(pred_val, color=self.pred_color, label=pred_label, linestyle='--', linewidth=1.7)
            ax.plot(ml_pred_val, color=self.ml_pred_color, label=ml_pred_label, linestyle="--", linewidth=1.7)
            ax.set_title(f"Ticker: {ticker}", fontsize=12)
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.grid(alpha=0.3)
            ax.legend(loc='best')

        # Hide unused subplots if n_plots is not a perfect multiple of cols
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis('off')
        
        self.fig = fig
    
    def show(self):
        if self.fig is not None:
            plt.show()
        else:
            raise ValueError("Figure not created!!")
        
class ResidPlot:
    def __init__(self, title, res_color='green'):
        self.title = title
        self.res_color = res_color
        self.fig = None

    def plot(self, data_dict, cols=3, figsize_per_plot=(5, 4)):
        """
        data_dict: dict { "TICKER": (true, pred) }
        cols: Number of columns in the grid
        """
        n_plots = len(data_dict)
        rows = math.ceil(n_plots / cols)

        fig, axes = plt.subplots(
            rows, cols, 
            figsize=(cols * figsize_per_plot[0], rows * figsize_per_plot[1]),
            constrained_layout=True
        )
        fig.suptitle(self.title, fontsize=16, fontweight='bold')

        if isinstance(axes, np.ndarray):
            axes_flat = axes.flatten()
        else:
            axes_flat = [axes]

        for i, (ticker, (true_val, pred_val)) in enumerate(data_dict.items()):
            ax = axes_flat[i]
            
            # Plotting lines
            ax.plot(true_val - pred_val, color=self.res_color, label='True', linewidth=1.5)
            
            ax.set_title(f"Ticker: {ticker}", fontsize=12)
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.grid(alpha=0.3)
            ax.legend(loc='best')

        # Hide unused subplots if n_plots is not a perfect multiple of cols
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis('off')
        
        self.fig = fig

        
    def show(self):
        if self.fig is not None:
            plt.show()
        else:
            raise ValueError("Figure not created!!")    

class MultiLineResidPlot:
    def __init__(self, title, colors):
        self.title = title
        self.colors = colors
        self.fig = None

    def plot(self, data_dict, figsize):
        fig, ax = plt.subplots(figsize= figsize)
        
        for i, (ticker, (true_val, pred_val)) in enumerate(data_dict.items()):
            ax.plot(true_val - pred_val, color=self.colors[i], label=ticker)  
            ax.legend()
        ax.set_title(self.title)
        ax.set_xlabel("Time")    
        ax.set_ylabel("Residuals")
        self.fig = fig


    def show(self):
        if self.fig is not None:
            plt.show()
        else:
            raise ValueError("No Figure Created!!!")    

class MultiLinePlot:
    def __init__(self, title, colors):
        self.title = title
        self.colors = colors
        self.fig = None
        
    def plot(self, true_val, data_dict, figsize):
        fig, ax = plt.subplots(figsize=figsize)

        plt.plot(true_val, color="black", label="True")
        for i, (ticker, (_, pred_val)) in enumerate(data_dict.items()):
            ax.plot(pred_val, color=self.colors[i], label=ticker)  
            ax.legend()
        
        ax.set_title(self.title)
        ax.set_xlabel("")
        self.fig = fig
    
    def show(self):
        if self.fig is not None:
            self.fig.show()
        else:
            raise ValueError("No Figure Created!!!")    


class BacktestEquityPlot:
    def __init__(self, title, colors="black"):
        self.title = title
        self.colors = colors
        self.fig = None
    
    def plot(self, curves:dict, figsize=(10,6)):
        fig, ax = plt.subplots(figsize=figsize)

        for i, (model, curve) in enumerate(curves.items()):
            ax.plot(curve, color=self.colors[i], label=model)
            ax.legend()

        ax.set_title(self.title)
        ax.set_xlabel("")
        self.fig = fig
    
    def show(self):
        if self.fig is not None:
            plt.show()
        else:
            raise ValueError("No Figure created !!!")
            



class BacktestEquityDrawdownPlot:
    def __init__(self, title, colors=None):
        self.title = title
        # Default colors if none are provided
        self.colors = colors if colors else ["#000000", "#ab560b", "#2ca02c", "#b92020", "#9467bd", "#E7C60A"]
        self.fig = None
    
    def plot(self, curves: dict, figsize=(10, 8)):
        """
        Expects 'curves' to be a dictionary where:
        key = model_name (str)
        value = tuple(equity_curve, drawdown_curve)
        """
        # Create a 2-panel chart, sharing the X-axis. 
        # The equity curve gets 75% of the vertical space (height_ratio 3:1)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True, 
                                       gridspec_kw={'height_ratios': [3, 1]})

        for i, (model, (equity, drawdown)) in enumerate(curves.items()):
            color1 = self.colors[(i % len(self.colors)) + 1]
            color2 = self.colors[(i % len(self.colors)) + 1]
            color3 = "#5C5959"
            # --- 1. Top Panel: Equity Curve ---
            ax1.plot(equity, color=color1, label=model, linewidth=1.5)
            
            # --- 2. Bottom Panel: Drawdown ("Underwater" Chart) ---
            # Plot the line and fill the area up to 0 for that classic underwater look
            ax2.plot(drawdown, color=color2, linewidth=1)
            ax2.fill_between(range(len(drawdown)), drawdown, 0, color=color3, alpha=0.3)

        # Formatting Top Panel (Equity)
        ax1.set_title(self.title, fontsize=14, fontweight='bold')
        ax1.set_ylabel("Portfolio Value", fontsize=10)
        ax1.legend(loc="upper left")
        ax1.grid(True, linestyle='--', alpha=0.5)

        # Formatting Bottom Panel (Drawdown)
        ax2.set_title("Drawdown", fontsize=10)
        ax2.set_ylabel("Drop from Peak", fontsize=10)
        ax2.set_xlabel("Time (Steps)", fontsize=10)
        ax2.yaxis.set_major_formatter(PercentFormatter(1.0)) # Formats -0.1 as -10%
        ax2.grid(True, linestyle='--', alpha=0.5)

        plt.tight_layout()
        self.fig = fig
    
    def show(self):
        if self.fig is not None:
            plt.show()
        else:
            raise ValueError("No Figure created !!! Call plot() first.")
        
