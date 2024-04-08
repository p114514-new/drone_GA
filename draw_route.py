from collections import defaultdict

from matplotlib import pyplot as plt
from tqdm import tqdm


def multi_draw():
    file_path = 'Agg.txt'
    clusters = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            x, y, index = map(float, line.strip().split())
            clusters[int(index)].append((x, y))
    # 画出塔的位置
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'pink']

    plt.figure(figsize=(10, 8))

    # for index, points in clusters.items():
    #     x_values, y_values = zip(*points)
    #     plt.scatter(x_values[0], y_values[0], marker='v',
    #                 c=colors[index % len(colors)], s=150)
    #     plt.scatter(x_values[1:], y_values[1:], label=f'Cluster {index + 1}', c=colors[index % len(colors)])
    seeds = [191, 192, 193, 194, 195, 196, 197, 198, 199]
    file_names = ['res_pop_200/'+str(seeds)+'Agg__200_with0.txt' for seeds in seeds]
    # 将文件190Agg.txt中的路线画到图上（每一行是每一个聚类的路线）
    for img_idx, file_name in tqdm(enumerate(file_names)):
        # 在一个图片上绘制九个子图
        plt.subplot(3, 3, img_idx + 1)
        with open(file_name, 'r') as file:
            for index, line in enumerate(file):
                # 去掉[和]
                line = line[1:-2]
                # 绘制路径
                color_index = 0
                path = list(map(int, line.split(',')))
                x_values, y_values = zip(*[clusters[index][i] for i in path])

                # 当每个0出现后，换一种颜色
                for i in range(len(path) - 1):
                    if path[i] == 0:
                        color_index += 1
                    plt.plot([x_values[i], x_values[i + 1]], [y_values[i], y_values[i + 1]], c=colors[color_index % len(colors)])
                # 设置子图的title
                plt.title(f'routes for seed={seeds[img_idx]}, epoch=2000')
    # 子图坐标轴不重复
    plt.tight_layout()
    plt.legend()

    plt.grid(True)
    plt.show()

def single_draw():
    file_path = 'Agg.txt'
    clusters = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            x, y, index = map(float, line.strip().split())
            clusters[int(index)].append((x, y))
    # 画出塔的位置
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'purple', 'orange', 'pink']

    plt.figure(figsize=(10, 8))

    for index, points in clusters.items():
        x_values, y_values = zip(*points)
        plt.scatter(x_values[0], y_values[0], marker='v',
                    c=colors[0], s=150)
        plt.scatter(x_values[1:], y_values[1:], label=f'Cluster {index + 1}', c=colors[index % len(colors)])

    # 将文件190Agg.txt中的路线画到图上（每一行是每一个聚类的路线）
    with open('res_pop_200/199Agg__200_with0.txt', 'r') as file:
        for index, line in enumerate(file):
            # 去掉[和]
            line = line[1:-2]
            # 绘制路径
            color_index = 0
            path = list(map(int, line.split(',')))
            x_values, y_values = zip(*[clusters[index][i] for i in path])

            # 当每个0出现后，换一种颜色
            for i in range(len(path) - 1):
                if path[i] == 0:
                    color_index += 1
                plt.plot([x_values[i], x_values[i + 1]], [y_values[i], y_values[i + 1]], c=colors[color_index % len(colors)])



    plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Drone Clusters and Routes')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    multi_draw()