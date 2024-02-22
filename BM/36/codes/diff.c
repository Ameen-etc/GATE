#include <stdio.h>
#include <math.h>

double x_n_h(double h, double x_prev) {
    double a = (1 + h/2) * x_prev - (h/400) * x_prev * x_prev;
    double x_n = (-(1 - h/2) + sqrt((1 - h/2)*(1 - h/2) + h * a / 100)) / (h/200);
    return x_n;
}

int main() {
    double x0 = 100; // Initial value of x
    double h = 0.1;  // Step size
    int num_steps = 1000; // Number of steps
    double nh[num_steps + 1]; // Array to store nh values
    double x_values[num_steps + 1]; // Array to store corresponding x(nh) values

    // Compute and store x(nh) values
    double x = x0;
    for (int n = 0; n <= num_steps; n++) {
        nh[n] = n * h;
        x_values[n] = x;
        x = x_n_h(h, x);
    }

    // Write nh and x(nh) values to a .dat file
    FILE *fp = fopen("discrete_data.dat", "w");
    if (fp == NULL) {
        printf("Error opening file.\n");
        return 1;
    }

    for (int i = 0; i <= num_steps; i++) {
        fprintf(fp, "%.6f %.6f\n", nh[i], x_values[i]);
    }

    fclose(fp);
    printf("Data written to discrete_data.dat.\n");

    return 0;
}

