"""
Author: Johnson
Time：2023-10-08 22:18
"""
import logging
import math
import random
from collections import defaultdict
from matplotlib import pyplot as plt
import numpy as np
from tqdm import tqdm


# 计算两座塔之间的距离
def get_distance(tower1, tower2):
    return ((tower1[0] - tower2[0]) ** 2 + (tower1[1] - tower2[1]) ** 2) ** 0.5


def get_dist_to_other_points(data):
    dist_list = []
    for index, points in data.items():
        dist_to_other_points = []
        x_values, y_values = list(zip(*points))
        for i in range(len(x_values)):
            dist_to_other_points.append(get_distance((x_values[i], y_values[i]), (x_values[0], y_values[0])))
        dist_list.append(dist_to_other_points)
    return dist_list


class GA():
    def __init__(self, towers, population_size=200):
        self.number_of_charge_boxes = 10  # 充电桩的数量
        self.drone_maxdist = 2700 * 10  # 无人机最大飞行距离
        self.population_size = population_size  # 种群数量
        self.towers = towers  # 塔的位置坐标
        self.crossover_rate = 0.8
        self.number_of_generations = 5000
        self.mutation_rate = 0.01
        self.mutation_rate2 = 0.05
        self.mutation_rate3 = 0.2

    # 初始化种群
    def initialize_population(self):
        population_size = self.population_size
        # 确定np.random.shuffle的随机种子
        number_of_charge_boxes = self.number_of_charge_boxes
        population = []
        for i in range(population_size):
            chromosome = []
            # 打乱path的index，第一座塔不变，加入到population中
            chromosome.append(0)
            path = np.arange(1, len(self.towers))
            np.random.shuffle(path)
            for p in path:
                chromosome.append(p)
            # print(f"{i}:{chromosome}")
            population.append(chromosome)
        return population

    def get_dist(self, i, j):
        return get_distance(self.towers[i], self.towers[j])

    # 适应度函数
    def fitness(self, path):
        battery = self.drone_maxdist
        towers = self.towers
        distance = 0
        charge_times = 0
        i = 0
        new_path = []
        new_path.append(0)
        while i < len(path) - 1:
            dist_between_towers = self.get_dist(path[i], path[i + 1])
            if i + 1 < len(path) and battery - dist_between_towers - \
                    self.get_dist(path[i + 1], path[0]) < 0:
                new_path.append(0)
                new_path.append(path[i + 1])
                charge_times += 1
                battery = self.drone_maxdist - self.get_dist(path[0], path[i + 1])
                distance = distance + self.get_dist(path[0], path[i]) + \
                           self.get_dist(path[0], path[i + 1])
            elif battery - dist_between_towers < 0:
                # print("battery error")
                if battery - self.get_dist(path[i], path[0]) < 0:
                    return float('inf'), charge_times, new_path
                distance = distance + self.get_dist(path[i], path[0]) + self.get_dist(path[0], path[i + 1])
                new_path.append(path[0])
                new_path.append(path[i + 1])
                battery = self.drone_maxdist - self.get_dist(path[0], path[i + 1])
                charge_times += 1
            else:
                distance += dist_between_towers
                new_path.append(path[i + 1])
                battery -= self.get_dist(path[i], path[i + 1])
            i += 1

        distance += self.get_dist(path[0], path[-1])
        battery -= self.get_dist(path[0], path[-1])
        if battery < 0:
            # print("battery error")
            return float('inf'), charge_times, new_path
        new_path.append(0)
        # print(f"{i}->{0}: distance: {distance}  battery: {battery}")
        return distance, charge_times, new_path

    def crossover(self, parent1, parent2, parent3):
        n = len(parent1)
        offspring = [None] * n

        # Step 1: Choose a random starting city from parent 1
        current_city = parent1[0]
        offspring[0] = current_city

        # Step 3: Find the shortest distance from current city to next city
        for i in range(1, n):
            # find the next city from parents according to current city
            next_city1 = parent1[(parent1.index(current_city) + 1) % n]
            next_city2 = parent2[(parent2.index(current_city) + 1) % n]
            next_city3 = parent3[(parent3.index(current_city) + 1) % n]

            # Step 3: Determine whether to be greedy or not
            if np.random.random() < self.crossover_rate:
                # Step 4.1: Select the shortest distance
                distances = [
                    (current_city, next_city1),
                    (current_city, next_city2),
                    (current_city, next_city3)
                ]
                shortest_distance = min(distances, key=lambda x: get_distance(self.towers[x[0]], self.towers[x[1]]))
                next_city = shortest_distance[1]

                # Check for duplicate cities
                if next_city in offspring[:i]:
                    # Try to select the city from another parent
                    alternative_parents = [p for p in [parent1, parent2, parent3] if p[i] != current_city]
                    for parent in alternative_parents:
                        if parent[i] not in offspring:
                            next_city = parent[i]
                            break
                    else:
                        # Select a random city that is not among the selected ones
                        remaining_cities = [c for c in range(0, n) if c not in offspring]
                        next_city = random.choice(remaining_cities)

                offspring[i] = next_city
                current_city = next_city
            else:
                # Step 4.2: Select the next city randomly
                remaining_cities = [c for c in range(0, n) if c not in offspring]
                next_city = random.choice(remaining_cities)
                offspring[i] = next_city
                current_city = next_city

        return offspring

    # 变异操作
    def mutation(self, chromosome):
        # 如果概率小于变异率，就进行变异操作
        if np.random.random() < self.mutation_rate:
            for _ in range(5):
                mutation_point1 = np.random.randint(1, len(chromosome) - 1)
                mutation_point2 = np.random.randint(1, len(chromosome) - 1)
                new_chromosome = chromosome.copy()
                new_chromosome[mutation_point1], new_chromosome[mutation_point2] = chromosome[mutation_point2], \
                    chromosome[
                        mutation_point1]
                if self.fitness(new_chromosome)[0] < self.fitness(chromosome)[0]:
                    chromosome = new_chromosome
                    break
        return chromosome

    def mutation2(self, chromosome):
        new_chromosome = chromosome.copy()
        # 如果概率小于变异率，就进行变异操作
        if np.random.random() < self.mutation_rate3:
            for _ in range(10):
                # for i in range(1, len(chromosome) // 2):
                for i in range(1, len(chromosome) - 2):
                    if np.random.random() < self.mutation_rate2:
                        mutation_point = np.random.randint(i, len(chromosome) - 1)
                        new_chromosome[mutation_point], new_chromosome[i] = new_chromosome[i], new_chromosome[
                            mutation_point]
                if self.fitness(new_chromosome)[0] < self.fitness(chromosome)[0]:
                    chromosome = new_chromosome
                    break
        return chromosome

    # 选择操作
    def selection(self, population):
        fitness_list = []
        new_population = []
        for path in population:
            fitness_list.append([self.fitness(path), path])
        # 根据第一个元素对fitness_list 进行排序
        fitness_list.sort(key=lambda x: x[0])

        # 选择适应度前30%的个体
        for i in range(int(len(population) / 10 * 3)):
            new_population.append(fitness_list[i][1])

        # 构造population个新的子代
        for i in range(len(population)):
            parent1 = fitness_list[i][1]
            parent2 = fitness_list[(i + 1) % len(population)][1]
            parent3 = fitness_list[(i + 2) % len(population)][1]
            offspring = self.crossover(parent1, parent2, parent3)
            new_population.append(offspring)

        # 重新计算适应度并取前0.95 * population个个体
        fitness_list = []
        for path in new_population:
            fitness_list.append([self.fitness(path), path])
        fitness_list.sort(key=lambda x: x[0])
        new_population = [x[1] for x in fitness_list[:int(0.95 * self.population_size)]]

        # 向新population中添加随机个体直到population个
        while len(new_population) < self.population_size:
            chromosome = []
            chromosome.append(0)
            path = np.arange(1, len(new_population[0]))
            np.random.shuffle(path)
            for p in path:
                chromosome.append(p)
            new_population.append(chromosome)

        return new_population


def run(seed, file_path, population_size, total_epoch):
    # data为不同聚类中塔的位置坐标, 其中第一座塔为中心塔, 也是充电桩
    clusters = defaultdict(list)
    with open(file_path, 'r') as file:
        for line in file:
            x, y, index = map(float, line.strip().split())
            clusters[int(index)].append((x, y))

    # dist = get_dist_to_other_points(clusters)
    random.seed(seed)
    np.random.seed(seed)
    # 对于每一聚类的最小飞行距离、充电次数、最优路径
    best_of_cluster = defaultdict()

    # 对于每一聚类
    for index, towers in clusters.items():
        print(f"第{index + 1}个聚类" + 80 * "-")
        ga = GA(towers, population_size)
        population = ga.initialize_population()  # 初始化种群
        (shortest_distance, charge_times, new_path) = ga.fitness(population[0])  # 种群中第一个路径的适应度
        best_path = []
        for iteration in range(total_epoch):  # 迭代
            for i in range(1, len(population)):  # 寻找最优适应度（最短路径）
                dist, times, new_path = ga.fitness(population[i])
                if dist < shortest_distance:
                    shortest_distance = round(dist, 2)
                    charge_times = times
                    best_path = population[i]
                    best_new_path = new_path

            if (iteration + 1) % 500 == 0:
                # shortest_distance 保留两位小数

                print(f"iteration:{iteration + 1} shortest_distance:{shortest_distance}  charge_times:{charge_times}")
            # 选择操作
            population = ga.selection(population)
            # 变异操作
            for i in range(len(population)):
                population[i] = ga.mutation(population[i])
            if iteration >= 600:  # 有没有更智能的判断方法？
                for i in range(len(population) // 2, len(population)):
                    population[i] = ga.mutation2(population[i])
        best_path.append(0)
        print(best_path)
        print(best_new_path)
        # 更新最优路径
        best_of_cluster[index] = (shortest_distance, charge_times, best_path, best_new_path)

    # 计算总路程
    total_distance = 0
    total_charge = 0
    for index, towers in clusters.items():
        total_distance += best_of_cluster[index][0]
        total_charge += best_of_cluster[index][1]
    print(f"总路程：{total_distance}, 总充电次数：{total_charge}")

    return best_of_cluster, total_distance, total_charge


def multiseed():
    population_size = 200
    total_epoch = 2000
    logging.basicConfig(level=logging.INFO,  # 设置日志级别（INFO、WARNING、ERROR、CRITICAL）
                        format='%(asctime)s - %(levelname)s - %(message)s',  # 日志格式
                        filename='ga_route_popsize' + str(population_size) + '.txt',  # 日志文件
                        datefmt='%Y-%m-%d %H:%M')
    logger = logging.getLogger('ga_route_popsize' + str(population_size))
    file_path = 'Agg.txt'

    to_csv_file_path = 'agg_center_tower_result.csv'
    with open(to_csv_file_path, 'a') as file:
        file.write("seed,population_size,total_distance,total_charge,total_epoch\n")
    seeds = [190, 191, 192, 193, 194, 195, 196, 197, 198, 199]
    for seed in tqdm(seeds, desc='seeds'):  # 多次试验
        best_of_cluster, total_distance, total_charge = run(seed, file_path, population_size, total_epoch)
        logger.info(' ')
        logger.info(
            f"seed: {seed}, total_distance: {round(total_distance, 2)}, total_charge: {total_charge}, total_epoch: {total_epoch}, population_size: {population_size}")
        for idx in range(10):
            logger.info(f"第{idx + 1}个聚类" + 80 * "-")
            logger.info(f"最短路径：{round(best_of_cluster[idx][0], 2)}")
            logger.info(f"充电次数：{best_of_cluster[idx][1]}")
            logger.info(f"最优路径：{best_of_cluster[idx][2]}")
            logger.info(f"最优路径（包括充电站）：{best_of_cluster[idx][3]}")
            logger.info(80 * "-")
            # 将路线写进'res/' + {seed} + 'Agg.txt'中
            with open(f'res_pop_200/{seed}Agg_{population_size}.txt', 'a') as file:
                file.write(f"{best_of_cluster[idx][2]}\n")
            with open(f'res_pop_200/{seed}Agg__{population_size}_with0.txt', 'a') as file:
                file.write(f"{best_of_cluster[idx][3]}\n")

        # 将seed，total_epochs，total_distance, total_charge写入csv文件中
        with open(to_csv_file_path, 'a') as file:
            file.write(f"{seed},{population_size},{total_distance},{total_charge},{total_epoch}\n")


if __name__ == '__main__':
    multiseed()
