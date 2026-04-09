import pygame
import random
import sys
import time

# Initialize Pygame
pygame.init()

# Display & Game Settings
WIDTH, HEIGHT = 800, 600
BLOCK_SIZE = 20
BASE_SPEED = 10          # Starting FPS
MAX_SPEED = 25           # Speed cap to keep it playable
SPEED_INCREMENT = 0.5    # How much speed increases per segment

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
RED = (255, 0, 0)
GRID_COLOR = (50,50,50)

# Screen & Font Setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")
font = pygame.font.Font(None, 36)

def draw_grid():
    """Draws a subtle grid aligned to BLOCK_SIZE."""
    for x in range(0, WIDTH, BLOCK_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, BLOCK_SIZE):
        pygame.draw.line(screen, GRID_COLOR, (0, y), (WIDTH, y))

class SnakeGame:
    def __init__(self):
        self.snake = [(200, 200), (220, 200), (240, 200)]
        self.direction = "RIGHT"
        self.food = self.generate_food()
        self.score = 0
        self.start_time = time.time()

    def get_elapsed_time(self):
        """Returns elapsed time formatted as MM:SS"""
        elapsed = time.time() - self.start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        return f"{minutes:02d}:{seconds:02d}"

    def get_current_speed(self):
        """Increases speed as the snake grows longer, capped at MAX_SPEED."""
        # Starting length is 3, so we calculate increase from that baseline
        increase = (len(self.snake) - 3) * SPEED_INCREMENT
        return min(MAX_SPEED, BASE_SPEED + increase)

    def generate_food(self):
        while True:
            x = random.randint(0, (WIDTH - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            y = random.randint(0, (HEIGHT - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
            if (x, y) not in self.snake:
                return (x, y)

    def draw_game(self):
        screen.fill(BLACK)
        draw_grid()  # Grid drawn first
        
        # Draw snake & food
        for x, y in self.snake:
            pygame.draw.rect(screen, WHITE, (x, y, BLOCK_SIZE, BLOCK_SIZE))
        food_x, food_y = self.food
        pygame.draw.rect(screen, RED, (food_x, food_y, BLOCK_SIZE, BLOCK_SIZE))
        
        # UI Elements
        score_text = font.render(f"Score: {self.score}", True, WHITE)
        speed_text = font.render(f"Speed: {self.get_current_speed():.1f}x", True, WHITE)
        time_text = font.render(f"Time: {self.get_elapsed_time()}", True, WHITE)
        
        screen.blit(score_text, (10, 10))                  # Top-left
        screen.blit(speed_text, (WIDTH // 2 - 50, 10))     # Center
        screen.blit(time_text, (WIDTH - 150, 10))          # Top-right
        
        pygame.display.flip()

    def update_game(self):
        head = self.snake[-1]
        dx, dy = 0, 0
        if self.direction == "RIGHT": dx = BLOCK_SIZE
        elif self.direction == "LEFT": dx = -BLOCK_SIZE
        elif self.direction == "UP": dy = -BLOCK_SIZE
        elif self.direction == "DOWN": dy = BLOCK_SIZE

        new_head = (head[0] + dx, head[1] + dy)

        # Wall collision
        if not (0 <= new_head[0] < WIDTH and 0 <= new_head[1] < HEIGHT):
            self.game_over("Hit the wall!")
        # Self collision
        if new_head in self.snake[:-1]:
            self.game_over("Bit yourself!")

        self.snake.append(new_head)
        if new_head == self.food:
            self.score += 1
            self.food = self.generate_food()
        else:
            self.snake.pop(0)

    def game_over(self, reason):
        print(f"Game Over! {reason} Final Score: {self.score}")
        pygame.quit()
        sys.exit()

def main():
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

        game.update_game()
        game.draw_game()
        # Dynamically adjust frame rate based on snake length
        clock.tick(game.get_current_speed())

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()