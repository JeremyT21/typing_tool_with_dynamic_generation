import pygame
import time
import pandas as pd

'''
Handles the logic for showing the user the sentences and tracking their progress.
A lot of this code is machine-assisted, and should likely be credited !@chatgpt (Remember this Jeremy!)
'''

class TypingGame:

    def __init__(self):

        # init
        pygame.init()

        # Screen size / resolution
        self.screen_width = 1200
        self.screen_height = 800

        # Game label and whatnot
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("Adaptive Typing Trainer")

        self.font = pygame.font.SysFont("consolas", 32)

        self.margin_x = 40
        self.margin_y = 80
        self.line_spacing = 10

        self.cursor_visible = True
        self.cursor_timer = pygame.time.get_ticks()


    def reset(self, sentence):

        self.sentence = sentence
        self.typed = ""

        self.word_times = []
        self.mistypes = []

        self.current_word_start = time.time()
        self.current_errors = 0


    def layout_text(self):

        rows = []

        x = self.margin_x
        y = self.margin_y

        current_row = []

        for i, char in enumerate(self.sentence):

            text_surface = self.font.render(char, True, (255,255,255))
            char_width = text_surface.get_width()

            if x + char_width > self.screen_width - self.margin_x:

                rows.append(current_row)
                current_row = []

                x = self.margin_x
                y += text_surface.get_height() + self.line_spacing

            current_row.append((i, char, x, y))

            x += char_width

        if current_row:
            rows.append(current_row)

        return rows


    def draw(self):

        self.screen.fill((30,30,30))

        rows = self.layout_text()

        cursor_x = None
        cursor_y = None

        for row in rows:

            for i, char, x, y in row:

                if i < len(self.typed):

                    if self.typed[i] == char:
                        color = (0,200,0)
                    else:
                        color = (200,0,0)

                else:
                    color = (200,200,200)

                text = self.font.render(char, True, color)

                self.screen.blit(text, (x,y))

                if i == len(self.typed):
                    cursor_x = x
                    cursor_y = y

        # Having some issues with the enter, so this is a check for sentence completion.
        if self.typed == self.sentence:
            msg = self.font.render("Press ENTER for next sentence", True, (200, 200, 200))
            self.screen.blit(msg, (self.margin_x, self.margin_y + 700))

        # This adds the blinky cursor next to the letters.
        now = pygame.time.get_ticks()

        if now - self.cursor_timer > 500:
            self.cursor_visible = not self.cursor_visible
            self.cursor_timer = now


        if cursor_x is not None and self.cursor_visible:

            cursor_height = self.font.get_height()

            pygame.draw.rect(
                self.screen,
                (255,255,255),
                (cursor_x, cursor_y, 3, cursor_height)
            )

        pygame.display.flip()

    # This actually runs the game, and handles the enter to put a new sentence up.
    def run(self, sentence):

        self.reset(sentence)

        sentence_complete = False

        while True:

            self.draw()

            for event in pygame.event.get():

                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                if event.type == pygame.KEYDOWN:

                    # ENTER advances to next sentence
                    if sentence_complete and event.key == pygame.K_RETURN:
                        elapsed = (time.time() - self.current_word_start) * 1000

                        self.word_times.append(elapsed)
                        self.mistypes.append(self.current_errors)

                        words = self.sentence.split()

                        self.word_times = self.word_times[:len(words)]
                        self.mistypes = self.mistypes[:len(words)]

                        return pd.DataFrame({
                            "word": words,
                            "time_ms": self.word_times,
                            "mistypes": self.mistypes
                        })

                    if sentence_complete:
                        continue

                    if event.key == pygame.K_BACKSPACE:

                        self.current_errors += 1

                        if len(self.typed) > 0:
                            self.typed = self.typed[:-1]

                        continue

                    char = event.unicode

                    if len(self.typed) < len(self.sentence):

                        if char != self.sentence[len(self.typed)]:
                            self.current_errors += 1

                        self.typed += char

                        if char == " ":
                            elapsed = (time.time() - self.current_word_start) * 1000

                            self.word_times.append(elapsed)
                            self.mistypes.append(self.current_errors)

                            self.current_word_start = time.time()
                            self.current_errors = 0

            # detect completion
            if self.typed == self.sentence:
                sentence_complete = True