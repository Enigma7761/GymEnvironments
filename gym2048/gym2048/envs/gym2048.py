import pygame
import gym
from gym import spaces
import numpy as np
import random

pygame.init()

white, black, red, blue, green, orange = (255, 255, 255), (0, 0, 0), (255, 0, 0), (0, 0, 255), (0, 255, 0), (
    255, 100, 0)
colors = [(238, 228, 218), (237, 224, 200), (242, 177, 121), (245, 149, 99), (246, 124, 95), (246, 94, 59), (237, 207,
                                                                                                             114),
          (237, 204, 97), (237, 200, 80), (237, 197, 63), (237, 194, 46), (238, 228, 218, 0.35)]
tile_background = (205, 193, 180)
dark_text = (119, 110, 101)
borders = (187, 173, 160)
score = (157, 143, 130)
background = (250, 248, 239)
size = 80
font = pygame.font.SysFont('verdana', size * 3 // 8)
score_font = pygame.font.SysFont('verdana', size // 4)


class gym2048(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, width=4, height=4, size=size):
        super(gym2048, self).__init__()
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(1, 288), dtype=np.float32)
        self.height = height
        self.width = width
        self.size = size
        self.game_display = pygame.display.set_mode((width * size + size * 4, height * size + size * 4))
        self.board = [[square(0) for _ in range(self.height)] for _ in range(self.height)]
        self.score = 0
        self.last_move = None
        self.last_tile = None
        self.clock = pygame.time.Clock()

    def step(self, action):
        """
        0, up
        1, down
        2, left
        3, right
        """
        self.last_move = action
        action = np.squeeze(action).item()
        reward = 0
        moved = False
        num_moved = 0
        sum_log_tiles = 0
        if action == 0:
            for j in range(self.width):
                for i in range(1, self.height):
                    temp_tile = self.board[i][j].number
                    temp_move, temp_reward = self.move(i, j, 0)
                    reward += temp_reward
                    if temp_move:
                        moved = True
                        sum_log_tiles += np.log2(temp_tile)
                        num_moved += 1
        elif action == 1:
            for j in range(self.width):
                for i in range(self.height - 2, -1, -1):
                    temp_tile = self.board[i][j].number
                    temp_move, temp_reward = self.move(i, j, 1)
                    reward += temp_reward
                    if temp_move:
                        moved = True
                        sum_log_tiles += np.log2(temp_tile)
                        num_moved += 1
        elif action == 2:
            for i in range(self.height):
                for j in range(1, self.width):
                    temp_tile = self.board[i][j].number
                    temp_move, temp_reward = self.move(i, j, 2)
                    reward += temp_reward
                    if temp_move:
                        moved = True
                        sum_log_tiles += np.log2(temp_tile)
                        num_moved += 1
        elif action == 3:
            for i in range(self.height):
                for j in range(self.width - 2, -1, -1):
                    temp_tile = self.board[i][j].number
                    temp_move, temp_reward = self.move(i, j, 3)
                    reward += temp_reward
                    if temp_move:
                        moved = True
                        sum_log_tiles += np.log2(temp_tile)
                        num_moved += 1

        if moved:
            free_tiles = []
            for i in range(self.height):
                for j in range(self.width):
                    if self.board[i][j] == 0:
                        free_tiles.append((i, j))
            i, j = random.choice(free_tiles)
            self.board[i][j].number = 2 if np.random.rand() < 0.9 else 4
            self.last_tile = (i, j, self.board[i][j])
        self.reset_combined()
        self.score += reward
        done = self.check_if_done()
        if moved == True:
          reward = np.log2(reward)
          if np.isneginf(reward):
              reward = 0
          reward -= 0.2 * num_moved * sum_log_tiles
        # else:
        #   reward = -1
        state = self.np_obs()
        state = state.reshape(1, -1)
        state[state == 0] = 1
        state = np.log2(state)
        state = state.astype(int)
        state = np.eye(18)[state].reshape(1, -1)
        return state, reward, done, {"highest_tile": np.max(self.np_obs()), "score": self.score}

    def reset(self):
        self.board = [[square(0) for _ in range(self.height)] for _ in range(self.height)]
        self.board[np.random.randint(0, self.height)][
            np.random.randint(0, self.width)].number = 2 if np.random.rand() < 0.9 else 4
        self.score = 0
        state = self.np_obs()
        state = state.reshape(1, -1)
        state[state == 0] = 1
        state = np.log2(state)
        state = state.astype(int)
        state = np.eye(18)[state].reshape(1, -1)
        return state

    def render(self, mode='human', close=False):
        # Creating the board
        """
              j=0     j=w-1
        i=0   ###.....#
              ###.....#
              .........
        i=h-1 ###.....#
        """
        pygame.display.set_caption('2048')
        self.game_display.fill(background)
        pygame.draw.rect(self.game_display, score, [self.size * 2 + self.size * 3, self.size, self.size, self.size / 2])
        text = score_font.render(str(self.score), True, white)
        text_obj = text.get_rect()
        text_obj.center = (self.size * 5.5, self.size * 1.25)
        self.game_display.blit(text, text_obj)
        for i in range(self.height):
            for j in range(self.width):
                if self.board[i][j] == 0:
                    pygame.draw.rect(self.game_display, tile_background, [self.size * 2 + self.size * j, self.size * 2 + self.size * i, self.size, self.size])
                else:
                    color_ind = int((np.log(self.board[i][j].number) / np.log(2)) - 1)
                    pygame.draw.rect(self.game_display, colors[color_ind], [self.size * 2 + self.size * j, self.size * 2 + self.size * i, self.size, self.size])
                    text = font.render(str(self.board[i][j].number), True, white)
                    text_obj = text.get_rect()
                    text_obj.center = (self.size * (j + 2) + self.size//2, self.size * (i + 2) + self.size//2)
                    self.game_display.blit(text, text_obj)
                pygame.draw.rect(self.game_display, borders, [self.size * 2 + self.size * j, self.size * 2 + self.size * i, self.size, self.size], self.size // 20)
        pygame.draw.rect(self.game_display, borders, [self.size * 2, self.size * 2, self.height * self.size, self.width * self.size], self.size // 20)
        pygame.display.flip()
        pygame.event.get()
        self.clock.tick(20)

    def move(self, row, column, direction):
        if self.board[row][column] == 0:
            return False, 0
        if direction == 0:
            for i in range(row - 1, -1, -1):
                if self.board[i][column] == 0 and i > 0:
                    continue
                elif self.board[i][column] != 0:
                    if self.board[i][column] == self.board[row][column] and not self.board[i][column].combined:
                        reward = 2 * self.board[i][column].number
                        self.board[i][column].number *= 2
                        self.board[i][column].combined = True
                        self.board[row][column].number = 0
                        return True, reward
                    elif row != i + 1:
                        self.board[i + 1][column].number = self.board[row][column].number
                        self.board[row][column].number = 0
                        return True, 0
                    else:
                        return False, 0
                elif self.board[i][column] == 0 and i == 0:
                    self.board[0][column].number = self.board[row][column].number
                    self.board[row][column].number = 0
                    return True, 0
                else:
                    return False, 0
        if direction == 1:
            for i in range(row + 1, self.height):
                if self.board[i][column] == 0 and i < self.height - 1:
                    continue
                elif self.board[i][column] != 0:
                    if self.board[i][column] == self.board[row][column] and not self.board[i][column].combined:
                        reward = 2 * self.board[i][column].number
                        self.board[i][column].number *= 2
                        self.board[i][column].combined = True
                        self.board[row][column].number = 0
                        return True, reward
                    elif row != i - 1:
                        self.board[i - 1][column].number = self.board[row][column].number
                        self.board[row][column].number = 0
                        return True, 0
                    else:
                        return False, 0
                elif self.board[i][column] == 0 and i == self.height - 1:
                    self.board[self.height - 1][column].number = self.board[row][column].number
                    self.board[row][column].number = 0
                    return True, 0
                else:
                    return False, 0
        if direction == 2:
            for i in range(column - 1, -1, -1):
                if self.board[row][i] == 0 and i > 0:
                    continue
                elif self.board[row][i] != 0:
                    if self.board[row][i] == self.board[row][column] and not self.board[row][i].combined:
                        reward = 2 * self.board[row][i].number
                        self.board[row][i].number *= 2
                        self.board[row][i].combined = True
                        self.board[row][column].number = 0
                        return True, reward
                    elif column != i + 1:
                        self.board[row][i + 1].number = self.board[row][column].number
                        self.board[row][column].number = 0
                        return True, 0
                    else:
                        return False, 0
                elif self.board[row][i] == 0 and i == 0:
                    self.board[row][0].number = self.board[row][column].number
                    self.board[row][column].number = 0
                    return True, 0
                else:
                    return False, 0
        if direction == 3:
            for i in range(column + 1, self.width):
                if self.board[row][i] == 0 and i < self.width - 1:
                    continue
                elif self.board[row][i] != 0:
                    if self.board[row][i] == self.board[row][column] and not self.board[row][i].combined:
                        reward = 2 * self.board[row][i].number
                        self.board[row][i].number *= 2
                        self.board[row][i].combined = True
                        self.board[row][column].number = 0
                        return True, reward
                    elif column != i - 1:
                        self.board[row][i - 1].number = self.board[row][column].number
                        self.board[row][column].number = 0
                        return True, 0
                    else:
                        return False, 0
                elif self.board[row][i] == 0 and i == self.width - 1:
                    self.board[row][self.width - 1].number = self.board[row][column].number
                    self.board[row][column].number = 0
                    return True, 0
                else:
                    return False, 0

    def check_if_done(self):
        for i in range(0, self.width):
            for j in range(0, self.height):
                if i > 0:
                    if self.board[i - 1][j] == self.board[i][j]:
                        return False
                if i < self.width - 1:
                    if self.board[i + 1][j] == self.board[i][j]:
                        return False
                if j > 0:
                    if self.board[i][j - 1] == self.board[i][j]:
                        return False
                if j < self.height - 1:
                    if self.board[i][j + 1] == self.board[i][j]:
                        return False
                if self.board[i][j] == 0:
                    return False
        return True

    def reset_combined(self):
        for i in range(self.height):
            for j in range(self.width):
                self.board[i][j].combined = False

    def np_obs(self):
        ls = [[0 for i in range(self.width)] for i in range(self.height)]
        for i in range(self.height):
            for j in range(self.width):
                try:
                    ls[i][j] = self.board[i][j].number
                except AttributeError:
                    pass

        return np.array(ls)[:, :, np.newaxis]


class square:
    def __init__(self, number):
        self.number = number
        self.combined = False

    def __eq__(self, other):
        if isinstance(other, square):
            return self.number == other.number
        elif isinstance(other, int):
            return self.number == other
        return False


if __name__ == '__main__':
    env = gym2048()
    board = env.reset()
    print(board)
    done = False
    while True:
        _, board, done, _ = env.step(0)
        if done:
            env.reset()
        env.render()
        print(type(board))
        print()
        env.render()
        _, board, done, _ = env.step(1)
        if done:
            env.reset()
        print(type(board))
        print()
        env.render()
        _, board, done, _ = env.step(2)
        if done:
            env.reset()
        print(board)
        print()
        env.render()
        _, board, done, _ = env.step(3)
        if done:
            env.reset()
        print(board)
        print()
        env.render()
