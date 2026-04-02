from __future__ import annotations

from typing import Iterable, Optional

import pygame


class IntersectionRenderer:
    def __init__(self, width: int = 900, height: int = 700, fps: int = 30):
        pygame.init()
        self.width = width
        self.height = height
        self.fps = fps
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Kigali Traffic RL")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 24)
        self.small_font = pygame.font.SysFont("consolas", 18)

    def _light_colors(self, action: int):
        red = (190, 40, 40)
        green = (40, 180, 40)
        yellow = (210, 180, 40)

        if action == 0:
            return {"ns": green, "ew": red}
        if action == 1:
            return {"ns": red, "ew": green}
        return {"ns": yellow, "ew": yellow}

    def _draw_intersection(self):
        bg = (230, 231, 224)
        road = (52, 52, 56)
        lane = (220, 220, 220)

        self.screen.fill(bg)

        pygame.draw.rect(self.screen, road, pygame.Rect(self.width // 2 - 95, 0, 190, self.height))
        pygame.draw.rect(self.screen, road, pygame.Rect(0, self.height // 2 - 95, self.width, 190))

        pygame.draw.line(self.screen, lane, (self.width // 2, 0), (self.width // 2, self.height), 2)
        pygame.draw.line(self.screen, lane, (0, self.height // 2), (self.width, self.height // 2), 2)

    def _draw_lights(self, action: int):
        colors = self._light_colors(action)
        pygame.draw.circle(self.screen, colors["ns"], (self.width // 2 - 120, self.height // 2 - 120), 16)
        pygame.draw.circle(self.screen, colors["ns"], (self.width // 2 + 120, self.height // 2 + 120), 16)
        pygame.draw.circle(self.screen, colors["ew"], (self.width // 2 + 120, self.height // 2 - 120), 16)
        pygame.draw.circle(self.screen, colors["ew"], (self.width // 2 - 120, self.height // 2 + 120), 16)

    def _draw_queue_blocks(self, queues: Iterable[float]):
        q_n, q_s, q_e, q_w = [int(v) for v in queues]
        car_color = (66, 133, 244)

        for i in range(min(q_n, 18)):
            pygame.draw.rect(self.screen, car_color, pygame.Rect(self.width // 2 - 66, self.height // 2 - 120 - i * 18, 16, 12))
        for i in range(min(q_s, 18)):
            pygame.draw.rect(self.screen, car_color, pygame.Rect(self.width // 2 + 50, self.height // 2 + 105 + i * 18, 16, 12))
        for i in range(min(q_e, 18)):
            pygame.draw.rect(self.screen, car_color, pygame.Rect(self.width // 2 + 105 + i * 18, self.height // 2 - 66, 12, 16))
        for i in range(min(q_w, 18)):
            pygame.draw.rect(self.screen, car_color, pygame.Rect(self.width // 2 - 120 - i * 18, self.height // 2 + 50, 12, 16))

    def draw(self, queues, action: int, reward: float, step: int, mode_label: str = "RUN"):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self._draw_intersection()
        self._draw_lights(action)
        self._draw_queue_blocks(queues)

        title = self.font.render("Kigali Intersection Traffic Control", True, (32, 32, 32))
        info = self.small_font.render(
            f"mode={mode_label}  step={step}  action={action}  reward={reward:.2f}",
            True,
            (32, 32, 32),
        )
        queues_text = self.small_font.render(
            f"queues N,S,E,W: {int(queues[0])}, {int(queues[1])}, {int(queues[2])}, {int(queues[3])}",
            True,
            (32, 32, 32),
        )

        self.screen.blit(title, (20, 18))
        self.screen.blit(info, (20, 54))
        self.screen.blit(queues_text, (20, 80))

        pygame.display.flip()
        self.clock.tick(self.fps)
        return True

    def close(self):
        pygame.quit()
