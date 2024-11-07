import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

class ANIMATOR:
    def __init__(self, title, fps=60) -> None:
        self.title = title
        metadata = dict(title = self.title, artist='LLNN')
        self.writer = FFMpegWriter(fps, metadata=metadata)
        
    def img_video(self, list_of_data: list):
        fig = plt.figure(figsize=(10,4))
        with self.writer.saving(fig, f"{self.title}.mp4", 100):
            for data in list_of_data:
                plt.imshow(data)
                self.writer.grab_frame()
                
    def multiple_img(self, list_of_data: list, col, rows, cmap: str = 'viridis'):
        fig = plt.figure(figsize=(10,4))
        for i in range(1, col*rows+1):
            img = list_of_data[i-1]
            fig.add_subplot(rows, col, i)
            plt.axis('off') 
            plt.imshow(img, cmap=cmap)
        plt.show()
        
    def distribution_plot_video(self, Dis1, Dis2, name, cmap: str = 'viridis'):
        list_of_data = [Dis1, Dis2]
        fig = plt.figure(figsize=(10,4))
        with self.writer.saving(fig, f"{self.title}.mp4", 100):
            for i in range(len(Dis1)):
                for j in range(1,3):
                    fig.suptitle(f'iterations: {i}')
                    fig.add_subplot(1, 2, j)
                    plt.bar(name, list_of_data[j-1][i][0])
                self.writer.grab_frame()
                fig.clear()
        

# import numpy as np

# data = []
# for i in range(100):
#     data.append(np.random.randn(9,9))
    
# Animator = ANIMATOR(data, 10, 'random')

# Animator.img_plot()