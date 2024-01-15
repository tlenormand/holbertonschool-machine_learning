#!/usr/bin/env python3

from classes.Atari import Atari


if __name__ == '__main__':
    Atari_instance = Atari(
        game='Breakout-v5',
        load_model=True,
        epsilon=-1,
        epsilon_min=0.1,
        can_render=False,
        render_mode=None,  # None, human, rgb_array, single_rgb_array...
    )

    # Atari_instance.play()

    while True:
        try:
            Atari_instance.train()
        except KeyboardInterrupt:
            Atari_instance.quit()
            break
        except Exception as e:
            Atari_instance.Logs.error("While running main", e)
            Atari_instance.quit()
