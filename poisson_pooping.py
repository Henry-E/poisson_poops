import random
from itertools import count
import math
from copy import copy
from collections import Counter

from matplotlib import pyplot as plt
from matplotlib import patches as patches
from matplotlib.animation import FuncAnimation
import matplotlib.animation as animation

import numpy as np
from scipy.special import factorial

from tqdm import tqdm


# make a bird class which initializes a start and end point randomly
# And an update function which updates the location
# If the bird has reached its destination set a random new point on the map
# And a should i poop function that does a random check for pooping
# Then a separate function for taking the poop occurrence and adding it the
# list of occurences and checking its location against the car locations and
# putting down different coloured patches accordingly
class Bird:
    def __init__(self, max_visible_x, max_visible_y):
        # We're setting an arbitrary 4x the visible map for now
        range_multiplier = 4
        self.max_bird_x = max_visible_x * range_multiplier
        self.max_bird_y = max_visible_y * range_multiplier
        self.min_bird_x = - max_visible_x * (range_multiplier - 1)
        self.min_bird_y = - max_visible_y * (range_multiplier - 1)
        self.max_visible_x = max_visible_x
        self.max_visible_y = max_visible_y
        self.start_loc = self.random_non_visible_loc()
        self.end_loc = self.random_non_visible_loc()
        self.current_loc = copy(self.start_loc)
        self.speed = 1
        self.velocity = self.get_velocity()
        self.poop_rate = 0.05


    def random_non_visible_loc(self):
        # Generate x first
        while True:
            x_loc = random.randrange(self.min_bird_x, self.max_bird_x)
            if x_loc < -5 or self.max_visible_x + 5 < x_loc:
                break
        # Generate y
        while True:
            y_loc = random.randrange(self.min_bird_y, self.max_bird_y)
            if y_loc < -5 or self.max_visible_y + 5 < y_loc:
                break
        return (x_loc, y_loc)


    def did_it_poop(self):
        if random.random() < self.poop_rate:
            return True
        else:
            return False

    def get_velocity(self):
        vel_x =  self.end_loc[0] - self.start_loc[0]
        vel_y =  self.end_loc[1] - self.start_loc[1]
        normalization = math.sqrt(vel_x ** 2 + vel_y ** 2)
        if normalization < 0.5:
            velocity = self.get_velocity()
        else:
            velocity = [self.speed * vel_x / normalization, self.speed * vel_y / normalization]
        return velocity

    def update_loc(self):
        distance_to_target = math.sqrt(
            (self.end_loc[0] - self.current_loc[0])**2 +
            (self.end_loc[0] - self.current_loc[0])**2)
        if distance_to_target < 1:
            self.current_loc = copy(self.end_loc)
            self.start_loc = copy(self.end_loc)
            self.end_loc = self.random_non_visible_loc()
            self.velocity = self.get_velocity()
        else:
            self.current_loc = (self.current_loc[0] + self.velocity[0],
                                self.current_loc[1] + self.velocity[1])



def main():
    max_x = 80
    max_y = 80
    # Randomly instantiate the cars
    max_cars = 190
    cars = [(random.randrange(1, max_x - 3, 3), 
            random.randrange(1, max_y - 7, 13))
            for _ in range(max_cars)]
    cars = list(set(cars))
    num_cars = len(cars)

    # Get the poission points for the cars assuming one per car
    # We can do this cleaner eventually
    # https://stackoverflow.com/questions/51242748/plot-a-poisson-distribution-graph-in-python
    poisson_max_x = 6
    poisson_points = {'x': [], 'y': []}
    poisson_points['x'] = np.arange(0, poisson_max_x, 0.1)
    poisson_points['y'] = (np.exp(-1) * np.power(
        1, poisson_points['x'])) / factorial(poisson_points['x']) * num_cars
    poisson_max_y = math.ceil(max(poisson_points['y'])) + 1

    # Instantiate some birds
    birds = [Bird(max_x, max_y) for _ in range(90)]

    fig, axes = plt.subplots(1, 2, figsize=(12,6.75))
    index = count()


    car_poops = {'x': [], 'y': []}
    ground_poops = {'x': [], 'y': []}
    # TODO init this with 0 for all
    poops_per_car = Counter()
    poops_per_car.update({car: 0 for car in cars})
    # Have to use nonlocal command with integer variable
    # https://www.pythoncircle.com/post/680/solving-python-error-unboundlocalerror-local-variable-x-referenced-before-assignment/
    total_poops = 0
    def dynamic_frames():
        """A generator to have a variable number of frames

        https://stackoverflow.com/questions/48564181/how-to-stop-funcanimation-by-func-in-matplotlib/48564392#48564392
        """
        i = 0
        prev_num_poops = 0
        with tqdm(total=num_cars, smoothing=0.1) as t:
            # We want an average of one poop per car
            while total_poops <= num_cars:
                if prev_num_poops < total_poops:
                    t.update(total_poops - prev_num_poops)
                    prev_num_poops = total_poops
                i += 1
                yield i

    def anim(i):
        nonlocal total_poops
        axes[0].clear()
        axes[1].clear()
        # Remove numbers from edges
        axes[0].get_xaxis().set_visible(False)
        axes[0].get_yaxis().set_visible(False)
        # Add labels to poisson graph
        axes[1].set_ylabel('Number of cars')
        axes[1].set_xlabel('Number of poops')
        # Set limits
        axes[0].axis([0, max_x, -4, max_y - 4])
        # axes[1].axis([0, poisson_max_x, 0, poisson_max_y])
        # Testing out making y-axis much larger by showing all cars starting
        # off at 0
        axes[1].axis([-0.5, poisson_max_x, 0, num_cars])
        # Set shape
        axes[0].set_aspect('equal', 'box')
        # axes[1].set_aspect('auto', 'box')
        for car in cars:
            rect = patches.Rectangle(car, 2, 6,linewidth=1,edgecolor='b',facecolor='none')
            axes[0].add_patch(rect)
        for bird in birds:
            bird.update_loc()
            # I really don't like how it mixes up x and 0 so much. Like it's a
            # pain that one thing takes tuples as input and the other needs
            # lists, it seems pretty contradictory
            if bird.did_it_poop():
                for car in cars:
                    if (car[0] < bird.current_loc[0]
                            and bird.current_loc[0] < car[0] + 2) and (
                                car[1] < bird.current_loc[1]
                                and bird.current_loc[1] < car[1] + 6):
                        car_poops['x'].append(bird.current_loc[0])
                        car_poops['y'].append(bird.current_loc[1])
                        poops_per_car[car] += 1
                        total_poops += 1
                        break
                else:
                    ground_poops['x'].append(bird.current_loc[0])
                    ground_poops['y'].append(bird.current_loc[1])
            circle = patches.Circle(bird.current_loc, 1)
            axes[0].add_patch(circle)
        axes[0].scatter(ground_poops['x'], ground_poops['y'], c='y', s=0.5)
        axes[0].scatter(car_poops['x'], car_poops['y'], c='r', s=2)

        # Poisson dist of expected poops per car
        axes[1].plot(poisson_points['x'],
                     poisson_points['y'],
                     label='Predicted Poisson distribution with mean=1')
        # Not sure how well this will work but it's what we want to get
        x_values, y_values = zip(
            *Counter(list(poops_per_car.values())).items())
        axes[1].bar(x_values,
                    y_values,
                    color='b',
                    label='Observed poops per car')
        axes[1].legend(
            ['Predicted Poisson dist\nwith mean=1', 'Observed poops per car'])

    Writer = animation.writers['ffmpeg']
    writer = Writer(fps=25, metadata=dict(artist='Me'), bitrate=2600)
    # Don't forget to use repeat=False when using a generator
    ani = FuncAnimation(fig,
                        anim,
                        interval=200,
                        save_count=10000,
                        frames=dynamic_frames,
                        repeat=False)
    # fig.tight_layout()
    plt.tight_layout(pad=2.0)
    plt.show()
    # plt.gcf().subplots_adjust(bottom=0.9)
    # import ipdb; ipdb.set_trace()
    ani.save('poisson_poops_14th_april.mp4', writer=writer, dpi=240)

if __name__ == "__main__":
    main()
