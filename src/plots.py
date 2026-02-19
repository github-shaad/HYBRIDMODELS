import matplotlib.pyplot as plt
import numpy as np
import math

class MultiPlot:
    def __init__(self, title, true_color='blue', pred_color='red'):
        self.title = title
        self.true_color = true_color
        self.pred_color = pred_color

    def plot(self, data_dict, cols=3, figsize_per_plot=(5, 4)):
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
            ax.plot(true_val, color=self.true_color, label='True', linewidth=1.5)
            ax.plot(pred_val, color=self.pred_color, label='Pred', linestyle='--', linewidth=1.5)
            
            ax.set_title(f"Ticker: {ticker}", fontsize=12)
            ax.set_xlabel("Time")
            ax.set_ylabel("Price")
            ax.grid(alpha=0.3)
            ax.legend(loc='best')

        # Hide unused subplots if n_plots is not a perfect multiple of cols
        for j in range(i + 1, len(axes_flat)):
            axes_flat[j].axis('off')

        plt.show()

class ResidPlot:
    def __init__(self, title, res_color='green'):
        self.title = title
        self.res_color = res_color

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

        plt.show()

class MultiLineResidPlot:
    def __init__(self, title, colors):
        self.title = title
        self.colors = colors

    def plot(self, data_dict, figsize):
        plt.title(self.title)
        plt.figure(figsize=figsize)

        for i, (ticker, (true_val, pred_val)) in enumerate(data_dict.items()):
            plt.plot(true_val - pred_val, color=self.colors[i], label=ticker)  
            plt.legend()

        plt.show()

class MultiLinePlot:
    def __init__(self, title, colors):
        self.title = title
        self.colors = colors

    def plot(self, true_val, data_dict, figsize):
        plt.title(self.title)
        plt.figure(figsize=figsize)
        plt.plot(true_val, color="black", label="True")
        for i, (ticker, (_, pred_val)) in enumerate(data_dict.items()):
            plt.plot(pred_val, color=self.colors[i], label=ticker)  
            plt.legend()
        plt.show()

class MDDPlot:
    pass