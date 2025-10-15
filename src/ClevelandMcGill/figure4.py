import numpy as np
import os
import skimage.draw
import sys

sys.path.append('../')
from src.util import Util


class Figure4:
    SIZE = (100, 100)

    @staticmethod
    def generate_datapoint1():
        '''
        '''

        pairs = [10. * 10. ** ((i - 1.) / 12.) for i in range(1, 11)]
        value_A = np.random.choice(pairs)
        value_B = value_A
        while value_B == value_A:
            value_B = np.random.choice(pairs)

        data = [np.round(value_A), np.round(value_B)]

        if (value_A < value_B):
            ratio = np.round(value_A) / np.round(value_B)
        else:
            ratio = np.round(value_B) / np.round(value_A)

        label = ratio

        return data, label

    @staticmethod
    def generate_datapoint():
        """
        Generate two distinct values a, b from the Cleveland–McGill pair set,
        with an extra guard so that a + b <= 81. This ensures
        current_max = 93 - a - b >= 12 for data_to_type5.
        """
        pairs = [10. * 10. ** ((i - 1.) / 12.) for i in range(1, 11)]
        while True:
            value_A = np.round(np.random.choice(pairs))
            value_B = value_A
            while value_B == value_A:
                value_B = np.round(np.random.choice(pairs))

            # FIX: enforce enough headroom for data_to_type5
            # (i.e., 93 - a - b >= 12  ->  a + b <= 81)
            if (value_A + value_B) <= 81:
                break  # valid pair

        data = [value_A, value_B]
        ratio = (np.round(min(value_A, value_B)) / np.round(max(value_A, value_B)))
        label = ratio
        return data, label

    @staticmethod
    def data_to_type1(data):
        '''
        '''
        barchart = np.zeros((100, 100), dtype=bool)

        # now we need 8 more pairs
        all_values = [0] * 10
        all_values[0] = np.random.randint(10, 93)
        all_values[1] = data[0]  # fixed pos 1
        all_values[2] = data[1]  # fixed pos 2
        all_values[3] = np.random.randint(10, 93)
        all_values[4] = np.random.randint(10, 93)
        all_values[5] = np.random.randint(10, 93)
        all_values[6] = np.random.randint(10, 93)
        all_values[7] = np.random.randint(10, 93)
        all_values[8] = np.random.randint(10, 93)
        all_values[9] = np.random.randint(10, 93)

        start = 0
        for i, d in enumerate(all_values):

            if i == 0:
                start += 2
            elif i == 5:
                start += 8
            else:
                start += 0

            gap = 2
            b_width = 7

            left_bar = start + i * gap + i * b_width
            right_bar = start + i * gap + i * b_width + b_width

            # print left_bar, right_bar

            rr, cc = skimage.draw.line(99, left_bar, 99 - int(d), left_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(99, right_bar, 99 - int(d), right_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(99 - int(d), left_bar, 99 - int(d), right_bar)
            barchart[rr, cc] = 1

            if i == 1 or i == 2:
                # mark the max
                barchart[94:95, left_bar + b_width // 2:left_bar + b_width // 2 + 1] = 1

        return barchart

    @staticmethod
    def data_to_type3(data):
        '''
        '''
        barchart = np.zeros((100, 100), dtype=bool)

        # now we need 8 more pairs
        all_values = [0] * 10
        all_values[0] = np.random.randint(10, 93)
        all_values[1] = data[0]  # fixed pos 1
        all_values[2] = np.random.randint(10, 93)
        all_values[3] = np.random.randint(10, 93)
        all_values[4] = np.random.randint(10, 93)
        all_values[5] = np.random.randint(10, 93)
        all_values[6] = data[1]  # fixed pos 2
        all_values[7] = np.random.randint(10, 93)
        all_values[8] = np.random.randint(10, 93)
        all_values[9] = np.random.randint(10, 93)

        start = 0
        for i, d in enumerate(all_values):

            if i == 0:
                start += 2
            elif i == 5:
                start += 8
            else:
                start += 0

            gap = 2
            b_width = 7

            left_bar = start + i * gap + i * b_width
            right_bar = start + i * gap + i * b_width + b_width

            # print left_bar, right_bar

            rr, cc = skimage.draw.line(99, left_bar, 99 - int(d), left_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(99, right_bar, 99 - int(d), right_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(99 - int(d), left_bar, 99 - int(d), right_bar)
            barchart[rr, cc] = 1

            if i == 1 or i == 6:
                # mark the max
                barchart[94:95, left_bar + b_width // 2:left_bar + b_width // 2 + 1] = 1

        return barchart

    @staticmethod
    def data_to_type2(data):
        '''
        '''
        barchart = np.zeros((100, 100), dtype=bool)

        # we build the barchart to the top
        all_values = [0] * 10
        all_values[0] = data[0]  # fixed pos but max. 56
        current_max = 93 - all_values[0]
        all_values[1] = np.random.randint(8, current_max / 4.)

        all_values[2] = np.random.randint(8, current_max / 4.)

        all_values[3] = np.random.randint(8, current_max / 4.)

        all_values[4] = np.random.randint(8, current_max / 4.)

        current_max = np.sum(all_values[0:5])

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 10, 99 - int(current_max), 10)
        barchart[rr, cc] = 1

        rr, cc = skimage.draw.line(99, 40, 99 - int(current_max), 40)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values):

            rr, cc = skimage.draw.line(99 - (int(d) + current), 10, 99 - (int(d) + current), 40)
            barchart[rr, cc] = 1
            current += int(d)

            if i == 0:
                # mark the max
                barchart[99 - int(d) // 2:99 - int(d) // 2 + 1, 25:26] = 1

        all_values[5] = data[1]  # fixed pos but max. 56
        current_max = 93 - all_values[5]
        # print current_max
        all_values[6] = np.random.randint(8, current_max / 4.)

        all_values[7] = np.random.randint(8, current_max / 4.)

        all_values[8] = np.random.randint(8, current_max / 4.)

        all_values[9] = np.random.randint(8, current_max / 4.)

        current_max = np.sum(all_values[5:])

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 60, 99 - int(current_max), 60)
        barchart[rr, cc] = 1

        rr, cc = skimage.draw.line(99, 90, 99 - int(current_max), 90)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values[5:]):

            rr, cc = skimage.draw.line(99 - (int(d) + current), 60, 99 - (int(d) + current), 90)
            barchart[rr, cc] = 1
            current += int(d)

            if i == 0:
                # mark the max
                barchart[99 - int(d) // 2:99 - int(d) // 2 + 1, 75:76] = 1

        return barchart

    @staticmethod
    def data_to_type41(data):
        '''
        '''
        barchart = np.zeros((100, 100), dtype=bool)

        # we build the barchart to the top
        all_values = [0] * 10

        current_max = 93 - data[0]

        all_values[0] = np.random.randint(8, current_max / 4.)

        all_values[1] = np.random.randint(8, current_max / 4.)

        all_values[2] = np.random.randint(8, current_max / 4.)

        all_values[3] = np.random.randint(8, current_max / 4.)

        below_last_sum = np.sum(all_values[0:4])

        all_values[4] = data[0]

        current_max = np.sum(all_values[0:5])

        above_last_sum = current_max

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 10, 99 - int(current_max), 10)
        barchart[rr, cc] = 1

        rr, cc = skimage.draw.line(99, 40, 99 - int(current_max), 40)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values):

            rr, cc = skimage.draw.line(99 - (int(d) + current), 10, 99 - (int(d) + current), 40)
            barchart[rr, cc] = 1
            current += int(d)

            if i == 4:
                # mark the max
                # print current, d
                barchart[99 - current + (int(d) // 2):99 - current + (int(d) // 2) + 1, 25:26] = 1

        below_last_sum2 = below_last_sum
        above_last_sum2 = above_last_sum

        # print below_last_sum, above_last_sum

        ctr = 0

        while below_last_sum2 == below_last_sum or above_last_sum2 == above_last_sum:

            if ctr == 20:
                # this happens around 10 times in 100.000 samples.. so not really important
                raise Exception()

            current_max = 93 - data[1]

            all_values[5] = np.random.randint(8, current_max / 4.)

            all_values[6] = np.random.randint(8, current_max / 4.)

            all_values[7] = np.random.randint(8, current_max / 4.)

            all_values[8] = np.random.randint(8, current_max / 4.)

            below_last_sum2 = np.sum(all_values[5:9])

            all_values[9] = data[1]

            current_max = np.sum(all_values[5:])
            above_last_sum2 = current_max

            ctr += 1  # exception counter

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 60, 99 - int(current_max), 60)
        barchart[rr, cc] = 1

        rr, cc = skimage.draw.line(99, 90, 99 - int(current_max), 90)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values[5:]):

            rr, cc = skimage.draw.line(99 - (int(d) + current), 60, 99 - (int(d) + current), 90)
            barchart[rr, cc] = 1
            current += int(d)

            if i == 4:
                # mark the max
                # print current, d
                barchart[99 - current + (int(d) // 2):99 - current + (int(d) // 2) + 1, 75:76] = 1

        return barchart

    @staticmethod
    def data_to_type4(data):
        '''
        '''
        barchart = np.zeros((100, 100), dtype=bool)

        # we build the barchart to the top
        all_values = [0] * 10

        current_max = 93 - data[0]

        all_values[0] = np.random.randint(8, current_max / 4.)

        all_values[1] = np.random.randint(8, current_max / 4.)

        all_values[2] = np.random.randint(8, current_max / 4.)

        all_values[3] = np.random.randint(8, current_max / 4.)

        below_last_sum = np.sum(all_values[0:4])

        all_values[4] = data[0]

        current_max = np.sum(all_values[0:5])

        above_last_sum = current_max

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 10, 99 - int(current_max), 10)
        barchart[rr, cc] = 1

        rr, cc = skimage.draw.line(99, 40, 99 - int(current_max), 40)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values):

            rr, cc = skimage.draw.line(99 - (int(d) + current), 10, 99 - (int(d) + current), 40)
            barchart[rr, cc] = 1
            current += int(d)

            if i == 4:
                # mark the max
                # print current, d
                barchart[99 - current + (int(d) // 2):99 - current + (int(d) // 2) + 1, 25:26] = 1

        below_last_sum2 = below_last_sum
        above_last_sum2 = above_last_sum

        # print below_last_sum, above_last_sum

        # ctr = 0

        while below_last_sum2 == below_last_sum or above_last_sum2 == above_last_sum:
            # if ctr == 20:
            #     # this happens around 10 times in 100.000 samples.. so not really important
            #     raise Exception()

            current_max = 93 - data[1]

            all_values[5] = np.random.randint(8, current_max / 4.)

            all_values[6] = np.random.randint(8, current_max / 4.)

            all_values[7] = np.random.randint(8, current_max / 4.)

            all_values[8] = np.random.randint(8, current_max / 4.)

            below_last_sum2 = np.sum(all_values[5:9])

            all_values[9] = data[1]

            current_max = np.sum(all_values[5:])
            above_last_sum2 = current_max

            # ctr += 1  # exception counter

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 60, 99 - int(current_max), 60)
        barchart[rr, cc] = 1

        rr, cc = skimage.draw.line(99, 90, 99 - int(current_max), 90)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values[5:]):

            rr, cc = skimage.draw.line(99 - (int(d) + current), 60, 99 - (int(d) + current), 90)
            barchart[rr, cc] = 1
            current += int(d)

            if i == 4:
                # mark the max
                # print current, d
                barchart[99 - current + (int(d) // 2):99 - current + (int(d) // 2) + 1, 75:76] = 1

        return barchart

    @staticmethod
    def data_to_type51(data):
        '''
        '''
        barchart = np.zeros((100, 100), dtype=bool)

        # we build the barchart to the top
        all_values = [0] * 10

        current_max = 93 - data[0] - data[1]

        if current_max <= 9:
            # this won't work
            raise Exception('Out of bounds')

        all_values[0] = np.random.randint(3, current_max / 3.)

        all_values[1] = np.random.randint(3, current_max / 3.)

        all_values[2] = np.random.randint(3, current_max / 3.)

        all_values[3] = data[0]

        all_values[4] = data[1]

        current_max = np.sum(all_values[0:5])

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 10, 99 - int(current_max), 10)
        barchart[rr, cc] = 1

        rr, cc = skimage.draw.line(99, 40, 99 - int(current_max), 40)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values):

            rr, cc = skimage.draw.line(99 - (int(d) + current), 10, 99 - (int(d) + current), 40)
            barchart[rr, cc] = 1
            current += int(d)

            if i == 3 or i == 4:
                # mark the max
                # print current, d
                barchart[99 - current + (int(d) // 2):99 - current + (int(d) // 2) + 1, 25:26] = 1

        current_max = 93

        all_values[5] = np.random.randint(8, current_max / 5.)

        all_values[6] = np.random.randint(8, current_max / 5.)

        all_values[7] = np.random.randint(8, current_max / 5.)

        all_values[8] = np.random.randint(8, current_max / 5.)

        all_values[9] = np.random.randint(8, current_max / 5.)

        current_max = np.sum(all_values[5:])

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 60, 99 - int(current_max), 60)
        barchart[rr, cc] = 1

        rr, cc = skimage.draw.line(99, 90, 99 - int(current_max), 90)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values[5:]):
            rr, cc = skimage.draw.line(99 - (int(d) + current), 60, 99 - (int(d) + current), 90)
            barchart[rr, cc] = 1
            current += int(d)

        return barchart

    @staticmethod
    def data_to_type5(data):
        """
        Stacked bars with two fixed segments (data[0], data[1]) and three random ones,
        then a separate 5-segment stack. Requires enough slack (a+b <= 81).
        """
        barchart = np.zeros((100, 100), dtype=bool)
        all_values = [0] * 10

        current_max = 93 - int(data[0]) - int(data[1])

        # FIX: stronger guard to ensure randint(high) >= 4
        if current_max < 12:
            raise Exception('Out of bounds')

        # FIX: randint upper bounds must be integer and > 3
        high_top = int(np.floor(current_max / 3.))
        if high_top <= 3:
            raise Exception('Out of bounds')

        all_values[0] = np.random.randint(3, high_top)
        all_values[1] = np.random.randint(3, high_top)
        all_values[2] = np.random.randint(3, high_top)

        all_values[3] = int(data[0])
        all_values[4] = int(data[1])

        current_max = int(np.sum(all_values[0:5]))

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 10, 99 - int(current_max), 10)
        barchart[rr, cc] = 1
        rr, cc = skimage.draw.line(99, 40, 99 - int(current_max), 40)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values[:5]):
            rr, cc = skimage.draw.line(99 - (int(d) + current), 10, 99 - (int(d) + current), 40)
            barchart[rr, cc] = 1
            current += int(d)
            if i in (3, 4):
                barchart[99 - current + (int(d) // 2):99 - current + (int(d) // 2) + 1, 25:26] = 1

        # Second stack (unchanged logic, but ensure integer high)
        current_max = 93
        high_right = int(np.floor(current_max / 5.))
        high_right = max(9, high_right)  # keep original min=8 logic safely above low
        all_values[5] = np.random.randint(8, high_right)
        all_values[6] = np.random.randint(8, high_right)
        all_values[7] = np.random.randint(8, high_right)
        all_values[8] = np.random.randint(8, high_right)
        all_values[9] = np.random.randint(8, high_right)

        current_max = int(np.sum(all_values[5:]))

        rr, cc = skimage.draw.line(99, 60, 99 - int(current_max), 60)
        barchart[rr, cc] = 1
        rr, cc = skimage.draw.line(99, 90, 99 - int(current_max), 90)
        barchart[rr, cc] = 1

        current = 0
        for d in all_values[5:]:
            rr, cc = skimage.draw.line(99 - (int(d) + current), 60, 99 - (int(d) + current), 90)
            barchart[rr, cc] = 1
            current += int(d)

        return barchart
