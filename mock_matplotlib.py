"""
Mock matplotlib module for demonstration purposes
Simulates matplotlib functionality without actual plotting
"""

class MockColormap:
    def __init__(self, name):
        self.name = name

class MockCM:
    viridis = MockColormap('viridis')
    plasma = MockColormap('plasma')
    YlOrRd = MockColormap('YlOrRd')

class MockPyplot:
    def __init__(self):
        self.cm = MockCM()
    
    def figure(self, figsize=None):
        print(f"Creating figure with size {figsize}")
        return MockFigure()
    
    def subplots(self, nrows=1, ncols=1, figsize=None):
        print(f"Creating {nrows}x{ncols} subplots with size {figsize}")
        fig = MockFigure()
        if nrows == 1 and ncols == 1:
            return fig, MockAxes()
        else:
            axes = [[MockAxes() for _ in range(ncols)] for _ in range(nrows)]
            return fig, axes
    
    def colorbar(self, im, ax=None, cax=None):
        print("Adding colorbar")
        return MockColorbar()
    
    def tight_layout(self):
        print("Applying tight layout")
    
    def savefig(self, filename, dpi=150, bbox_inches='tight'):
        print(f"Saving figure to {filename}")
    
    def close(self):
        print("Closing figure")

class MockFigure:
    def add_gridspec(self, nrows, ncols, hspace=None, wspace=None):
        print(f"Adding gridspec {nrows}x{ncols}")
        return MockGridSpec()
    
    def add_subplot(self, *args):
        print(f"Adding subplot {args}")
        return MockAxes()
    
    def subplots_adjust(self, **kwargs):
        print(f"Adjusting subplots: {kwargs}")
    
    def add_axes(self, rect):
        print(f"Adding axes at {rect}")
        return MockAxes()
    
    def suptitle(self, title, fontsize=None):
        print(f"Setting super title: {title}")
    
    def colorbar(self, im, cax=None):
        print("Adding figure colorbar")
        return MockColorbar()

class MockGridSpec:
    def __getitem__(self, key):
        return f"GridSpec[{key}]"

class MockAxes:
    def imshow(self, data, origin=None, cmap=None, vmin=None, vmax=None, aspect=None):
        print(f"Creating imshow plot with shape {getattr(data, 'shape', 'unknown')}")
        return MockImage()
    
    def plot(self, x, y, fmt=None, color=None, linewidth=None, label=None, markersize=None, 
             markeredgecolor=None, markeredgewidth=None):
        print(f"Plotting line with {len(x) if hasattr(x, '__len__') else 1} points")
        return [MockLine()]
    
    def fill_between(self, x, y1, y2, alpha=None, color=None):
        print("Creating fill_between plot")
    
    def bar(self, x, height, color=None):
        print(f"Creating bar plot with {len(x) if hasattr(x, '__len__') else 1} bars")
        return [MockBar() for _ in range(len(x) if hasattr(x, '__len__') else 1)]
    
    def set_xlabel(self, label):
        print(f"Setting x-label: {label}")
    
    def set_ylabel(self, label):
        print(f"Setting y-label: {label}")
    
    def set_title(self, title, pad=None):
        print(f"Setting title: {title}")
    
    def legend(self):
        print("Adding legend")
    
    def grid(self, visible=True, alpha=None):
        print(f"Setting grid: {visible}")
    
    def set_ylim(self, limits):
        print(f"Setting y-limits: {limits}")
    
    def text(self, x, y, text, fontsize=None, color=None, fontweight=None, ha=None, va=None):
        print(f"Adding text at ({x}, {y}): {text}")
    
    def axis(self, setting):
        print(f"Setting axis: {setting}")
    
    def table(self, cellText=None, cellLoc=None, loc=None, colWidths=None):
        print("Creating table")
        return MockTable()
    
    def flatten(self):
        return [self]

class MockImage:
    pass

class MockLine:
    pass

class MockBar:
    def get_x(self):
        return 0
    
    def get_width(self):
        return 1
    
    def get_height(self):
        return 1

class MockColorbar:
    def set_label(self, label, rotation=None, labelpad=None):
        print(f"Setting colorbar label: {label}")

class MockTable:
    def auto_set_font_size(self, auto):
        pass
    
    def set_fontsize(self, size):
        pass
    
    def scale(self, x, y):
        pass

class MockLinearSegmentedColormap:
    @staticmethod
    def from_list(name, colors):
        return MockColormap(name)

class MockColors:
    LinearSegmentedColormap = MockLinearSegmentedColormap

# Create mock numpy for array operations
class MockArray:
    def __init__(self, data):
        self.data = data
        if hasattr(data, '__len__'):
            if hasattr(data[0], '__len__'):
                self.shape = (len(data), len(data[0]))
            else:
                self.shape = (len(data),)
        else:
            self.shape = ()
    
    @property
    def T(self):
        return self

class MockNumpy:
    def array(self, data):
        return MockArray(data)

# Set up mock modules
import sys

class MockMatplotlib:
    def use(self, backend):
        print(f"Setting matplotlib backend: {backend}")
    
    pyplot = MockPyplot()
    colors = MockColors()

sys.modules['matplotlib'] = MockMatplotlib()

sys.modules['matplotlib.pyplot'] = MockPyplot()
sys.modules['matplotlib.colors'] = MockColors()
sys.modules['numpy'] = MockNumpy()

print("Mock matplotlib and numpy modules loaded")