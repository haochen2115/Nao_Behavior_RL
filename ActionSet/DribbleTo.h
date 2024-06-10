option(DribbleTo, (float) dribble_angle) {
  common_transition {
    // Common transitions if any
  }

  initial_state(start) {
    transition {
      goto dribbleto;
    }
    action {
      // Initial actions if any
    }
  }

  state(dribbleto) {
    transition {
      // Transitions if any
    }
    action {
      float dribble_target_x, dribble_target_y;
      dribble_target_x = ball.positionField.x() + 1000 * cos(dribble_angle);
      dribble_target_y = ball.positionField.y() + 1000 * sin(dribble_angle);
      Dribble(dribble_target_x, dribble_target_y);
    }
  }
}
