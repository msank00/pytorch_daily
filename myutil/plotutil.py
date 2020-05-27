import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

class PublishQualityPlot:
    def __init__(self,):
        self.params = {'xtick.labelsize': 16,
                        'ytick.labelsize': 16,
                        'font.size': 15,
                        'figure.autolayout': True,
                        'axes.titlesize' : 16,
                        'axes.labelsize' : 17,
                        'lines.linewidth' : 2,
                        'lines.markersize' : 6,
                        'legend.fontsize': 13,
                        }
    
    def set_params(self):
        pylab.rcParams.update(self.params)
        plt.rc('font', family='serif')
        plt.rc('axes', linewidth=1)
    
    def preferable_plot_style(self):
        style_list = ['seaborn-muted','seaborn-deep',
                      'seaborn-colorblind','seaborn-bright',
                      'seaborn-pastel']
        return style_list