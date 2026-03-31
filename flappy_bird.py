import gymnasium as gym
import flappy_bird_gymnasium
import pygame

# Create environment
env = gym.make("FlappyBird-v0", render_mode="human")

state, info = env.reset()
done = False

# Initialize pygame (important!)
pygame.init()

# Gym already creates a window, we just access it
screen = pygame.display.get_surface()

while not done:
    action = 0  # default: no flap

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                action = 1  # flap

    # Take step in environment
    state, reward, done, truncated, info = env.step(action)

    # Render environment
    env.render()

# Cleanup
env.close()
pygame.quit()