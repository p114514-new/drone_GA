//
// Created by swx on 2023/9/7.
//

#ifndef std_dev
#define std_dev 10
#endif

#ifndef DRONEGA_SUPPORT_H
#define DRONEGA_SUPPORT_H

#include<bits/stdc++.h>

using namespace std;
const double length_of_map = 60 * 1000, width_of_map = 60 * 1000;  //in meters
const int number_of_towers = 200;
vector<tuple<double, double, int>> coordinates(number_of_towers);

std::random_device rd;
std::mt19937 generator(rd());

double getRandomDouble(double upperBound) {
    std::uniform_real_distribution<double> distribution(0.0, upperBound);
    return distribution(generator);
}

void generate_points_near_dense_lines() {
    const int numLines = 8;  // Number of lines to generate
    const int numPointsPerLine = number_of_towers / numLines;  // Number of points per line

    // Calculate the distance between adjacent points
    double distanceBetweenPoints = length_of_map / (numPointsPerLine - 1);
    vector<pair<double, double>> Fpoints(0), Epoints(0);

    int index_ = 0;

    for (int line = 0; line < numLines; ++line) {
        if (getRandomDouble(1) > 0.5) {
            pair<double, double> Fpoint{0, getRandomDouble(width_of_map)}, Epoint{length_of_map,
                                                                                  getRandomDouble(width_of_map)};
            if (Epoint.first < Fpoint.first) swap(Fpoint, Epoint);
            Fpoints.emplace_back(Fpoint);
            Epoints.emplace_back(Epoint);
        } else {
            pair<double, double> Fpoint{getRandomDouble(length_of_map), 0}, Epoint{getRandomDouble(length_of_map),
                                                                                   width_of_map};
            if (Epoint.first < Fpoint.first) swap(Fpoint, Epoint);
            Fpoints.emplace_back(Fpoint);
            Epoints.emplace_back(Epoint);
        }
    }

    for (int line = 0; line < numLines; ++line) {
        vector<pair<double, double>> points;
        double distance_mean = 2000.0;
        double distance_std_dev = std::sqrt(2.5E5);
        double min_distance = 1000.0;
        double max_distance = 3000.0;

        double slope = (Epoints[line].second - Fpoints[line].second) / (Epoints[line].first - Fpoints[line].first);
        double distance;

        double x = Fpoints[line].first;
        double y = Fpoints[line].second;
        points.emplace_back(x, y);

        while (x < Epoints[line].first) {
            distance = std::max(std::min(std::normal_distribution<double>(distance_mean, distance_std_dev)(generator),
                                         max_distance), min_distance);
            double dx = distance / std::sqrt(1 + std::pow(slope, 2));
            double dy = slope * dx;

            x += dx;
            y += dy;

            points.emplace_back(x, y);
        }
        int start_index = std::uniform_int_distribution<int>(0, static_cast<int>(points.size()) - number_of_towers / 8)(
                generator);
        for (int i = start_index; i < start_index + number_of_towers / 8; i++){
            get<0>(coordinates[index_]) = points[i].first;
            get<1>(coordinates[index_]) = points[i].second;
            get<2>(coordinates[index_++]) = line;
        }
    }
}


void shuffleCoordinates() {
    std::shuffle(coordinates.begin(), coordinates.end(), generator);
}

void outputCoordinatesToFile(const std::string &filename) {
    std::ofstream outputFile(filename);
    if (outputFile.is_open()) {
        for (const auto &coordinate: coordinates) {
            outputFile << std::fixed << std::setprecision(2) << get<0>(coordinate) << " " << get<1>(coordinate) << " "
                       << get<2>(coordinate) << "\n";
        }
        outputFile.close();
        std::cout << "Coordinates have been successfully output to file: " << filename << std::endl;
    } else {
        std::cout << "Unable to open file: " << filename << std::endl;
    }
}

void outputDistanceMatrixToFile(const std::string &filename) {
    std::vector<std::vector<double>> distanceMatrix(number_of_towers, std::vector<double>(number_of_towers, 0.0));
    for (int i = 0; i < number_of_towers; ++i) {
        for (int j = 0; j < number_of_towers; ++j) {
            double dx = get<0>(coordinates[i]) - get<0>(coordinates[j]);
            double dy = get<1>(coordinates[i]) - get<1>(coordinates[j]);
            double modification;
            if (get<2>(coordinates[i]) == 1 && get<2>(coordinates[j]) == 1) modification = cos(5 * M_PI / 180);
            else modification = cos(20 * M_PI / 180);
            distanceMatrix[i][j] = std::sqrt(dx * dx + dy * dy) / modification;
        }
    }

    std::ofstream outputFile(filename);
    if (outputFile.is_open()) {
        for (const auto &row: distanceMatrix) {
            for (const auto &distance: row) {
                outputFile << std::fixed << std::setprecision(2) << distance << " ";
            }
            outputFile << "\n";
        }
        outputFile.close();
        std::cout << "Distance matrix has been successfully output to file: " << filename << std::endl;
    } else {
        std::cout << "Unable to open file: " << filename << std::endl;
    }
}

#endif //DRONEGA_SUPPORT_H
