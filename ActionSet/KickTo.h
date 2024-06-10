option(KickTo, (float) kick_angle) {
  common_transition {
    // Common transitions if any
  }

  initial_state(start) {
    transition {
      goto kickto;
    }
    action {
      // Initial actions if any
    }
  }

  state(kickto) {
    transition {
      // Transitions if any
    }
    action {
      Vector2f kick_target;
      kick_target.x() = ball.positionField.x() + 1000 * cos(kick_angle);
      kick_target.y() = ball.positionField.y() + 1000 * sin(kick_angle);
      Kick(kick_target);
    }
  }
}
