from numpy import *
from matplotlib import pyplot as plt

# Dataset
DATASET = 'dataset'

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        totalError += (y - (m * x + b)) ** 2
    return totalError / float(len(points))

def gradient_descent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    b = starting_b
    m = starting_m
    for _ in range(num_iterations):
        b, m = step_gradient(b, m, points, learning_rate)
    return [b, m]

def step_gradient(b_current, m_current, points, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = float(len(points))
    for i in range(len(points)):
        x = points[i, 0]
        y = points[i, 1]
        b_gradient += -(2 / N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2 / N) * x * (y - ((m_current * x) + b_current))
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]

def scrapper(filename): 
    # Collect data
    points = genfromtxt(filename + '.csv', delimiter=',', skip_header=1)
    return points

def show_plot(points, b, m):
    x = linspace(min(points[:, 0]), max(points[:, 0]), 200)
    y = m * x + b
    plt.plot(x, y, label=f'y = {m:.2f}x + {b:.2f}', color='magenta')
    plt.scatter(points[:, 0], points[:, 1], color='green', alpha=0.7, s=20)
    
    plt.title('Linear Regression Line and Data Points')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    
    plt.show()
    

def main(dataset):
    
    # Hyperparameters  
    LEARNING_RATE = 0.0001
    initial_b = 0
    initial_m = 0
    num_iterations = 100000
    points = scrapper(dataset)
    
    # Show plot
    show_plot(points, initial_b, initial_m)
    
    # Train model
    print(f"Starting gradient descent at b = {initial_b}, m = {initial_m}, error = {compute_error_for_line_given_points(initial_b, initial_m, points):.4f}")
    b, m = gradient_descent_runner(points, initial_b, initial_m, LEARNING_RATE, num_iterations)
    print(f"After {num_iterations} iterations: b = {b:.4f}, m = {m:.4f}, error = {compute_error_for_line_given_points(b, m, points):.4f}")
    
    # Show plot
    show_plot(points, b, m)
    
if __name__ == "__main__":
    main(DATASET)
