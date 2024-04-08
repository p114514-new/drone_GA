#include <bits/stdc++.h>
#include "support.h"

using namespace std;

// Structure to represent a point in 2D space
struct Point {
    double x;
    double y;
    int index;

    Point(double x, double y, int z) : x(x), y(y), index(z) {}

    Point() { index = x = y = 0; }
};

const int number_of_charge_boxes = 10;
const double dronemaxdist = 2700 * 10;
const double examinededuction = 480 * 10;
const int population_number = 50;
vector<Point> towers(number_of_towers), charge_box(number_of_charge_boxes);
vector<vector<double>> dist_matrix(number_of_towers, vector<double>(number_of_towers));
const double cluster_max_point_number_threshold = 1.3;
const double param1 = 0.6;
const double epsilon = 3000 / cos(20 * M_PI / 180) + DBL_MIN;

// Function to calculate the Euclidean distance between two points
double calculateDistance(const Point &p1, const Point &p2) {
    double dx = p1.x - p2.x;
    double dy = p1.y - p2.y;
    return sqrt(dx * dx + dy * dy);
}

// Function to assign each point to the nearest cluster centroid
void assignPointsToClusters(const vector<Point> &points, const vector<Point> &centroids, vector<int> &assignments,
                            const double threshold = cluster_max_point_number_threshold) {
    vector<int> cluster_count(centroids.size(), 0);
    for (int i = 0; i < points.size(); ++i) {
        double minDistance = numeric_limits<double>::max();
        int clusterIndex = 0;

        for (long j = 0; j < centroids.size(); ++j) {
            double distance = calculateDistance(points[i], centroids[j]);
            if (cluster_count[j] >= number_of_towers / static_cast<double>(centroids.size()) * threshold)
                distance = DBL_MAX;
            if (distance < minDistance) {
                minDistance = distance;
                clusterIndex = j;
            }
        }

        assignments[i] = clusterIndex;
        cluster_count[clusterIndex]++;
    }
}

// Function to update the cluster centroids based on the assigned points
void updateClusterCentroids(const vector<Point> &points, const vector<int> &assignments, vector<Point> &centroids) {
    vector<int> clusterSizes(centroids.size(), 0);
    vector<double> clusterSumsX(centroids.size(), 0.0);
    vector<double> clusterSumsY(centroids.size(), 0.0);

    for (size_t i = 0; i < points.size(); ++i) {
        int clusterIndex = assignments[i];
        clusterSizes[clusterIndex]++;
        clusterSumsX[clusterIndex] += points[i].x;
        clusterSumsY[clusterIndex] += points[i].y;
    }

    for (size_t i = 0; i < centroids.size(); ++i) {
        if (clusterSizes[i] > 0) {
            centroids[i].x = clusterSumsX[i] / clusterSizes[i];
            centroids[i].y = clusterSumsY[i] / clusterSizes[i];
        }
    }
}

// K-means clustering algorithm
auto kMeansClustering(const vector<Point> &points, int k, int maxIterations) {
    if (points.size() < k) {
        cout << "Error: Number of clusters is greater than the number of points!" << endl;
        exit(114514);
    }

    // Initialize centroids with the first k points
    vector<Point> centroids(points.begin(), points.begin() + k);

    // Assignments vector to store the index of the assigned centroid for each point
    vector<int> assignments(points.size(), 0);

    // Perform iterations until convergence or reaching the maximum number of iterations
    for (int iteration = 0; iteration < maxIterations; ++iteration) {
        // Assign points to the nearest cluster centroid
        assignPointsToClusters(points, centroids, assignments);

        // Update cluster centroids based on the assigned points
        updateClusterCentroids(points, assignments, centroids);
    }

    // Create clusters based on the final assignments
    vector<vector<Point>> clusters(k);
    for (size_t i = 0; i < points.size(); ++i) {
        int clusterIndex = assignments[i];
        clusters[clusterIndex].push_back(points[i]);
    }

    return centroids;
}

void readCoordinatesFromFile(const string &filename) {
    ifstream file(filename);

    if (!file) {
        cout << "Error: Failed to open file." << endl;
        return;
    }

    string line;
    for (int i = 0; i < number_of_towers; ++i) {
        getline(file, line);
        istringstream iss(line);
        double x, y, z;
        if (iss >> x >> y >> z) {
            towers[i].x = x;
            towers[i].y = y;
        }
    }

    file.close();
}

void readDistanceMatrixFromFile(const string &filename) {
    ifstream file(filename);

    if (!file) {
        cout << "Error: Failed to open file." << endl;
        return;
    }

    string line;
    for (int i = 0; i < number_of_towers; ++i) {
        getline(file, line);
        istringstream iss(line);
        double x;
        for (int j = 0; j < number_of_towers; ++j) {
            iss >> x;
            dist_matrix[i][j] = x;
        }
    }

    file.close();
}

void outputClusterToFile(vector<vector<Point>> clusters, const string &filename) {
    std::ofstream outputFile(filename);
    if (outputFile.is_open()) {
        for (int i = 0; i < clusters.size(); ++i) {
            for (const auto &point: clusters[i]) {
                outputFile << std::fixed << std::setprecision(2) << point.x << " " << point.y << " "
                           << i << "\n";
            }
        }
        outputFile.close();
        std::cout << "Clusters have been successfully output to file: " << filename << std::endl;
    } else {
        std::cout << "Unable to open file: " << filename << std::endl;
    }
}

vector<vector<Point>> reassemblyPoints(std::vector<Point> &centroids) {
    vector<vector<Point>> clusters(centroids.size());
    vector<bool> assigned(towers.size(), false);
    unordered_map<string, int> cache;         // maps a point to its cluster index


    // Repeat the initial process for several times first
    for (int k = 0; k < static_cast<double>(towers.size()) / static_cast<double>(centroids.size()) * param1; ++k) {
        for (int i = 0; i < centroids.size(); ++i) {
            Point &centroid = centroids[i];
            double minDistance = std::numeric_limits<double>::max();
            int closestTowerIndex = -1;

            for (int j = 0; j < towers.size(); ++j) {
                Point &tower = towers[j];
                if (!assigned[j]) {
                    double distance = calculateDistance(centroid, tower);
                    if (distance < minDistance) {
                        minDistance = distance;
                        closestTowerIndex = j;
                    }
                }
            }

            if (closestTowerIndex != -1) {
                assigned[closestTowerIndex] = true;
                clusters[i].emplace_back(towers[closestTowerIndex]);
                cache[to_string(towers[closestTowerIndex].x) + to_string(towers[closestTowerIndex].y)] = i;
            } else exit(114);
        }
    }

    // Assign remaining free points
    bool allPointsAssigned = int(param1 + DBL_MIN);

    while (!allPointsAssigned) {
        allPointsAssigned = true;

        for (int i = 0; i < number_of_towers; ++i) {
            if (assigned[i]) continue;

            Point tower_x = towers[i];

            for (int j = 0; j < number_of_towers; ++j) {
                if (!assigned[j]) continue;
                Point tower_y = towers[j];
                if (dist_matrix[i][j] <= epsilon) {
                    assigned[i] = true;
                    clusters[cache[to_string(tower_y.x) + to_string(tower_y.y)]].emplace_back(tower_x);
                    cache[to_string(tower_x.x) + to_string(tower_x.y)] = cache[to_string(tower_y.x) +
                                                                               to_string(tower_y.y)];
                    break;
                }
            }

            if (!assigned[i]) {
                allPointsAssigned = false;
            }
        }
    }

    return clusters;
}

vector<Point> GA(vector<Point> &cluster) {
    int n = cluster.size();

}

int main() {
    // Generate your vector of points here
    readCoordinatesFromFile("coordinates.txt");
    readDistanceMatrixFromFile("distance_matrix.txt");

    int numClusters = 10;
    int maxIterations = 100;

    auto centroids = kMeansClustering(towers, numClusters, maxIterations);

    auto clusters = reassemblyPoints(centroids);

    outputClusterToFile(clusters, "clusters.txt");

    for (auto &cluster: clusters) {
        cout << cluster.size() << endl;
    }
}
