import pygame
import random
import sys
import time
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import argparse

# Initialize Pygame
pygame.init()

# Display & Game Settings
WIDTH, HEIGHT = 600, 400
BLOCK_SIZE = 20
BASE_SPEED = 10          # Starting FPS
MAX_SPEED = 100          # Speed cap for AI training
SPEED_INCREMENT = 0.5    # How much speed increases per segment

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
GRID_COLOR = (50, 50, 50)

# Screen & Font Setup
screen = None
font = None

def init_pygame():
    global screen, font
    if screen is None:
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Snake Game RL")
        font = pygame.font.Font(None, 36)

def draw_grid():
    """Draws a subtle grid aligned to BLOCK_SIZE."""
    for x in range(0, WIDTH, BLOCK_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, BLOCK_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y))

class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(WIDTH // 2, HEIGHT // 2), 
                      (WIDTH // 2 - BLOCK_SIZE, HEIGHT // 2), 
                      (WIDTH // 2 - 2 * BLOCK_SIZE, HEIGHT // 2)]
        self.direction = "RIGHT"
        self.score = 0
        self.food = self.generate_food()
        self.start_time = time.time()
        self.frame_iteration = 0
        return self.get_state()

    def get_state(self):
        head = self.snake[-1]
        
        # Danger detection (1 step ahead)
        point_l = (head[0] - BLOCK_SIZE, head[1])
        point_r = (head[0] + BLOCK_SIZE, head[1])
        point_u = (head[0], head[1] - BLOCK_SIZE)
        point_d = (head[0], head[1] + BLOCK_SIZE)

        dir_l = self.direction == "LEFT"
        dir_r = self.direction == "RIGHT"
        dir_u = self.direction == "UP"
        dir_d = self.direction == "DOWN"

        state = [
            # Danger straight
            (dir_r and self.is_collision(point_r)) or 
            (dir_l and self.is_collision(point_l)) or 
            (dir_u and self.is_collision(point_u)) or 
            (dir_d and self.is_collision(point_d)),

            # Danger right
            (dir_u and self.is_collision(point_r)) or 
            (dir_d and self.is_collision(point_l)) or 
            (dir_l and self.is_collision(point_u)) or 
            (dir_r and self.is_collision(point_d)),

            # Danger left
            (dir_d and self.is_collision(point_r)) or 
            (dir_u and self.is_collision(point_l)) or 
            (dir_r and self.is_collision(point_u)) or 
            (dir_l and self.is_collision(point_d)),
            
            # Move direction
            dir_l, dir_r, dir_u, dir_d,
            
            # Food location 
            self.food[0] < head[0],  # food left
            self.food[0] > head[0],  # food right
            self.food[1] < head[1],  # food up
            self.food[1] > head[1]   # food down
        ]

        return np.array(state, dtype=int)

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.snake[-1]
        # Wall collision
        if pt[0] >= WIDTH or pt[0] < 0 or pt[1] >= HEIGHT or pt[1] < 0:
            return True
        # Self collision
        if pt in self.snake[:-1]:
            return True
        return False

    def generate_food(self):
        while True:
            x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            if (x, y) not in self.snake:
                return (x, y)

    def step(self, action):
        """
        Action: 0=Straight, 1=Right Turn, 2=Left Turn
        """
        self.frame_iteration += 1
        # 1. Update Direction
        clock_wise = ["RIGHT", "DOWN", "LEFT", "UP"]
        idx = clock_wise.index(self.direction)

        if action == 0: # Straight
            new_dir = clock_wise[idx]
        elif action == 1: # Right turn
            new_idx = (idx + 1) % 4
            new_dir = clock_wise[new_idx]
        else: # Left turn
            new_idx = (idx - 1) % 4
            new_dir = clock_wise[new_idx]

        self.direction = new_dir

        # 2. Move
        head = self.snake[-1]
        dx, dy = 0, 0
        if self.direction == "RIGHT": dx = BLOCK_SIZE
        elif self.direction == "LEFT": dx = -BLOCK_SIZE
        elif self.direction == "UP": dy = -BLOCK_SIZE
        elif self.direction == "DOWN": dy = BLOCK_SIZE

        new_head = (head[0] + dx, head[1] + dy)
        self.snake.append(new_head)

        # 3. Check Game Over
        reward = 0
        done = False
        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            done = True
            reward = -10
            return reward, done, self.score

        # 4. Check Food
        if new_head == self.food:
            self.score += 1
            reward = 10
            self.food = self.generate_food()
        else:
            self.snake.pop(0)
            # Small reward for staying alive / moving towards food
            reward = 0 

        return reward, done, self.score

    def draw_game(self, fps=None):
        init_pygame()
        # Handle events to keep window responsive
        pygame.event.pump()
        
        screen.fill(BLACK)
        draw_grid()
        
        for i, (x, y) in enumerate(self.snake):
            color = GREEN if i == len(self.snake) - 1 else WHITE
            pygame.draw.rect(screen, color, (x, y, BLOCK_SIZE, BLOCK_SIZE))
            
        pygame.draw.rect(screen, RED, (self.food[0], self.food[1], BLOCK_SIZE, BLOCK_SIZE))
        
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        screen.blit(score_text, (10, 10))
        
        if fps is not None:
            fps_text = font.render(f"FPS: {fps} (UP/DOWN to change, 0: Max, 1: 30)", True, WHITE)
            screen.blit(fps_text, (10, HEIGHT - 40))
            
        pygame.display.flip()

class SnakeEnv(gym.Env):
    def __init__(self, render_mode=None, fps=30):
        super(SnakeEnv, self).__init__()
        self.game = SnakeGame()
        self.action_space = spaces.Discrete(3) # 0: Straight, 1: Right, 2: Left
        self.observation_space = spaces.Box(low=0, high=1, shape=(11,), dtype=np.intc)
        self.render_mode = render_mode
        self.fps = fps
        self.clock = pygame.time.Clock()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        state = self.game.reset()
        return state, {}

    def step(self, action):
        reward, done, score = self.game.step(action)
        state = self.game.get_state()
        if self.render_mode == "human":
            self.render()
        return state, reward, done, False, {"score": score}

    def render(self):
        # Dynamic FPS control
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    self.fps += 10
                elif event.key == pygame.K_DOWN:
                    self.fps = max(1, self.fps - 10)
                elif event.key == pygame.K_0:
                    self.fps = 1000 # "Max" speed
                elif event.key == pygame.K_1:
                    self.fps = 30   # Reset to default
                    
        self.game.draw_game(fps=self.fps)
        self.clock.tick(self.fps)

def main():
    parser = argparse.ArgumentParser(description="Snake Game with RL")
    parser.add_argument("mode", nargs="?", choices=["train", "play"], help="Mode to run the game in")
    parser.add_argument("--visualize", action="store_true", help="Visualize training")
    parser.add_argument("--fps", type=int, default=30, help="Tick rate for visualization (default: 30)")
    parser.add_argument("--continue", dest="continue_train", action="store_true", help="Continue training from the last saved model")
    args = parser.parse_args()

    if args.mode == "train":
        import train
        train.train(visualize=args.visualize, fps=args.fps, continue_training=args.continue_train)
    elif args.mode == "play":
        import play
        play.play()
    else:
        # Normal Human Play
        init_pygame()
        clock = pygame.time.Clock()
        game = SnakeGame()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_UP and game.direction != "DOWN":
                        game.direction = "UP"
                    elif event.key == pygame.K_DOWN and game.direction != "UP":
                        game.direction = "DOWN"
                    elif event.key == pygame.K_LEFT and game.direction != "RIGHT":
                        game.direction = "LEFT"
                    elif event.key == pygame.K_RIGHT and game.direction != "LEFT":
                        game.direction = "RIGHT"

            # Map absolute direction to "Straight" for the simplified step function
            # Since step() now takes relative actions, human play needs a slightly different loop or we adapt step()
            # For simplicity in human mode, we'll bypass the relative step() and use direct movement.
            
            # Re-implementing human logic for direct movement
            head = game.snake[-1]
            dx, dy = 0, 0
            if game.direction == "RIGHT": dx = BLOCK_SIZE
            elif game.direction == "LEFT": dx = -BLOCK_SIZE
            elif game.direction == "UP": dy = -BLOCK_SIZE
            elif game.direction == "DOWN": dy = BLOCK_SIZE

            new_head = (head[0] + dx, head[1] + dy)
            if game.is_collision(new_head):
                print(f"Game Over! Final Score: {game.score}")
                running = False
                continue

            game.snake.append(new_head)
            if new_head == game.food:
                game.score += 1
                game.food = game.generate_food()
            else:
                game.snake.pop(0)

            game.draw_game()
            clock.tick(BASE_SPEED + (game.score // 5))

        pygame.quit()

if __name__ == "__main__":
    main()