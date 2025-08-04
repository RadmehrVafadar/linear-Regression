from numpy import *

def compute_error_for_line_given_points(b, m, points):
    totalError = 0
    # for every point
    for i in range(0, len(points)):
        # get x value
        x = points[i, 0]
        y = points[i, 1]

        totalError += (y - (m*x + b)) ** 2

    # return the average
    return (totalError/float(len(points)))

# time for the magic baby
def step_gradient(b_current, m_current, points, learning_rate):
    N = float(len(points))
    b_gradient = 0
    m_gradient = 0
    
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # direction with respect to b and m
        # computing partial derivatives of our error function
        b_gradient += -(2/N) * (y - ((m_current * x) + b_current))
        m_gradient += -(2/N) * x * ( y - (m_current * x) + b_current)
    
    # update our b and m values using this partial derivatives
    new_b = b_current - (learning_rate * b_gradient)
    new_m = m_current - (learning_rate * m_gradient)
    return [new_b, new_m]


def gradient_decent_runner(points, starting_b, starting_m, learning_rate, num_iterations):
    # starting b and m
    b = starting_b
    m = starting_m

    # gradient decent
    for i in range(num_iterations):
        # update b and m with the new more accurate b and  m by performing 
        # this gradient step
        b, m = step_gradient(b, m, array(points), learning_rate)
    return b, m

def run():
    # Step 1 - collect our data
    points = genfromtxt('data.csv', delimiter=',')

    learning_rate = 0.0001
    # this is y= mx+b
    initial_b = 0
    initial_m = 0
    num_iterations = 1000

    # step 3 - train our model
    print(f'starting gradient decent at b={initial_b}, m={initial_m}, error={compute_error_for_line_given_points(initial_b, initial_m, points)}')
    [b, m] = gradient_decent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
    print(f'ending gradient decent at b={b}, m={m}, error={compute_error_for_line_given_points(b, m, points)}')


    
if __name__ == "__main__":
    run()