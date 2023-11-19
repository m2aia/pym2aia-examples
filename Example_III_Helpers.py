import matplotlib.pyplot as plt
import seaborn as sns

def showimg(image, cmap=None, title=None, size=(10,10), spacing=None, fig=None, subplot_pos=(1,1,1)):
    sns.set_theme(style="ticks")
    
    if fig is None:
        fig = plt.figure(figsize = size) # create a 5 x 5 figure 
    
    ax = fig.add_subplot(subplot_pos[0],subplot_pos[1],subplot_pos[2])

    ax.tick_params(
        which='both',
        bottom=False,
        left=False,
        labelleft=False,
        labelbottom=False)
    x = [10,60]
    y = [10,10]

    if spacing is not None:
        ax.plot(x, y, color="white", linewidth=3)
        ax.text(x[0], y[0]+7, f"{int(spacing[0]*(x[1]-x[0])* 1000)} μm", color="white",size=14)
        
    if title:
        ax.text(x[0], y[0]-3, title, color="white", size=14)
        
    im = ax.imshow(image, interpolation='none', cmap=cmap)
    fig.colorbar(im)

    return fig, ax 